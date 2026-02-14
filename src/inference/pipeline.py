"""
FRLM Inference Pipeline — end-to-end generation with interleaved retrieval.

Wraps the FRLMModel.generate() method with:
- Input tokenization and validation
- Structured response with retrieved facts & citations
- Router decision logging per token
- Fallback behaviour when retrieval confidence < threshold
- Batch inference support

Public API
----------
- ``InferencePipeline(model, tokenizer, faiss_index, kg_client, config)``
- ``generate(prompt, max_length, temperature, ...) → FRLMResponse``
- ``generate_batch(prompts, ...) → list[FRLMResponse]``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# Response containers
# ===========================================================================


@dataclass
class RetrievedFact:
    """A single fact retrieved during generation."""

    fact_id: str = ""
    subject_label: str = ""
    relation_type: str = ""
    object_label: str = ""
    confidence: float = 0.0
    temporal_mode: str = "CURRENT"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "fact_id": self.fact_id,
            "subject": self.subject_label,
            "relation": self.relation_type,
            "object": self.object_label,
            "confidence": round(self.confidence, 4),
            "temporal_mode": self.temporal_mode,
        }
        if self.valid_from:
            d["valid_from"] = self.valid_from
        if self.valid_to:
            d["valid_to"] = self.valid_to
        if self.source:
            d["source"] = self.source
        return d


@dataclass
class RouterDecision:
    """Router decision at a single generation step."""

    step: int = 0
    probability: float = 0.0
    decision: str = "generation"  # "retrieval" or "generation"
    num_facts_retrieved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "probability": round(self.probability, 4),
            "decision": self.decision,
            "num_facts_retrieved": self.num_facts_retrieved,
        }


@dataclass
class FRLMResponse:
    """Complete response from the inference pipeline."""

    generated_text: str = ""
    token_ids: Optional[List[int]] = None
    retrieved_facts: List[RetrievedFact] = field(default_factory=list)
    router_decisions: List[RouterDecision] = field(default_factory=list)
    retrieval_fraction: float = 0.0
    generation_fraction: float = 0.0
    num_tokens_generated: int = 0
    inference_time_ms: float = 0.0
    prompt: str = ""

    @property
    def num_retrieval_steps(self) -> int:
        return sum(1 for d in self.router_decisions if d.decision == "retrieval")

    @property
    def num_generation_steps(self) -> int:
        return sum(1 for d in self.router_decisions if d.decision == "generation")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "num_tokens_generated": self.num_tokens_generated,
            "retrieval_fraction": round(self.retrieval_fraction, 4),
            "generation_fraction": round(self.generation_fraction, 4),
            "retrieved_facts": [f.to_dict() for f in self.retrieved_facts],
            "router_decisions": [d.to_dict() for d in self.router_decisions],
            "inference_time_ms": round(self.inference_time_ms, 2),
        }


# ===========================================================================
# Inference Pipeline
# ===========================================================================


class InferencePipeline:
    """FRLM inference pipeline with interleaved retrieval and generation.

    Parameters
    ----------
    model : FRLMModel
        Trained FRLM model.
    tokenizer : Any
        Tokenizer compatible with the backbone (e.g. GPT2Tokenizer).
    faiss_index : Any, optional
        FAISS vector index for retrieval.
    kg_client : Any, optional
        Neo4j KG client for temporal fact filtering.
    config : InferenceConfig, optional
        Inference configuration.
    device : str
        Device for model inference.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        faiss_index: Any = None,
        kg_client: Any = None,
        config: Any = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.faiss_index = faiss_index
        self.kg_client = kg_client
        self.device = device

        # Extract config parameters with defaults
        if config is not None:
            self.max_length = getattr(config, "max_length", 512)
            self.temperature = getattr(config, "temperature", 0.7)
            self.top_k = getattr(config, "top_k", 50)
            self.top_p = getattr(config, "top_p", 0.95)
            self.do_sample = getattr(config, "do_sample", True)
            self.repetition_penalty = getattr(config, "repetition_penalty", 1.1)
            self.router_threshold = getattr(config, "router_threshold", 0.5)
        else:
            self.max_length = 512
            self.temperature = 0.7
            self.top_k = 50
            self.top_p = 0.95
            self.do_sample = True
            self.repetition_penalty = 1.1
            self.router_threshold = 0.5

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "InferencePipeline created: device=%s, max_length=%d, "
            "temperature=%.2f, router_threshold=%.2f",
            device,
            self.max_length,
            self.temperature,
            self.router_threshold,
        )

    @classmethod
    def from_config(
        cls,
        model: Any,
        tokenizer: Any,
        faiss_index: Any = None,
        kg_client: Any = None,
        config: Any = None,
    ) -> "InferencePipeline":
        """Factory from a full FRLMConfig.

        Parameters
        ----------
        model : FRLMModel
            Trained FRLM model.
        tokenizer : Any
            Tokenizer.
        faiss_index : Any, optional
            FAISS index.
        kg_client : Any, optional
            KG client.
        config : FRLMConfig, optional
            Full FRLM config (will use .inference section).
        """
        inf_config = getattr(config, "inference", None) if config else None
        device = getattr(inf_config, "device", "cpu") if inf_config else "cpu"
        return cls(
            model=model,
            tokenizer=tokenizer,
            faiss_index=faiss_index,
            kg_client=kg_client,
            config=inf_config,
            device=device,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        router_threshold: Optional[float] = None,
        timestamp: Any = None,
    ) -> FRLMResponse:
        """Generate text with interleaved retrieval.

        Parameters
        ----------
        prompt : str
            Input prompt text.
        max_length : int, optional
            Override default max generation length.
        temperature : float, optional
            Override default temperature.
        top_k : int, optional
            Override default top-k.
        top_p : float, optional
            Override default top-p.
        router_threshold : float, optional
            Override default router threshold.
        timestamp : date, optional
            Timestamp for AT_TIMESTAMP temporal mode.

        Returns
        -------
        FRLMResponse
            Complete response with text, facts, and metadata.
        """
        start_time = time.time()

        _max_length = max_length or self.max_length
        _temperature = temperature or self.temperature
        _top_k = top_k or self.top_k
        _top_p = top_p or self.top_p
        _threshold = router_threshold or self.router_threshold

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Generate using FRLMModel.generate()
        gen_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=_max_length - input_ids.size(1),
            temperature=_temperature,
            top_k=_top_k,
            top_p=_top_p,
            faiss_index=self.faiss_index,
            kg_client=self.kg_client,
        )

        # Parse model output
        generated_ids = gen_output["token_ids"]
        raw_facts = gen_output.get("retrieved_facts", [])
        raw_decisions = gen_output.get("router_decisions", [])

        # Decode tokens (skip prompt tokens)
        prompt_len = input_ids.size(1)
        new_token_ids = generated_ids[0, prompt_len:].tolist()
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # Convert facts to structured format
        retrieved_facts = self._convert_facts(raw_facts)

        # Convert router decisions
        router_decisions = self._convert_decisions(raw_decisions, _threshold)

        # Compute fractions
        total_steps = len(router_decisions) if router_decisions else 1
        ret_steps = sum(1 for d in router_decisions if d.decision == "retrieval")
        gen_steps = total_steps - ret_steps

        elapsed_ms = (time.time() - start_time) * 1000

        response = FRLMResponse(
            generated_text=generated_text,
            token_ids=new_token_ids,
            retrieved_facts=retrieved_facts,
            router_decisions=router_decisions,
            retrieval_fraction=ret_steps / total_steps if total_steps else 0.0,
            generation_fraction=gen_steps / total_steps if total_steps else 1.0,
            num_tokens_generated=len(new_token_ids),
            inference_time_ms=elapsed_ms,
            prompt=prompt,
        )

        logger.info(
            "Generated %d tokens (%.1fms): %d retrieval, %d generation steps",
            response.num_tokens_generated,
            elapsed_ms,
            ret_steps,
            gen_steps,
        )
        return response

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        router_threshold: Optional[float] = None,
    ) -> List[FRLMResponse]:
        """Generate text for a batch of prompts.

        Note: Due to the autoregressive nature with interleaved retrieval,
        batch processing is done sequentially per prompt.

        Parameters
        ----------
        prompts : list of str
            Input prompts.
        max_length : int, optional
            Override max length.
        temperature : float, optional
            Override temperature.
        top_k : int, optional
            Override top-k.
        top_p : float, optional
            Override top-p.
        router_threshold : float, optional
            Override router threshold.

        Returns
        -------
        list of FRLMResponse
        """
        responses: List[FRLMResponse] = []
        for i, prompt in enumerate(prompts):
            logger.debug("Processing prompt %d/%d: %s...", i + 1, len(prompts), prompt[:50])
            resp = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                router_threshold=router_threshold,
            )
            responses.append(resp)
        return responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_facts(self, raw_facts: List[Any]) -> List[RetrievedFact]:
        """Convert raw facts from model.generate() to structured format."""
        converted: List[RetrievedFact] = []
        for fact in raw_facts:
            if hasattr(fact, "fact_id"):
                # Pydantic Fact object
                converted.append(
                    RetrievedFact(
                        fact_id=getattr(fact, "fact_id", ""),
                        subject_label=getattr(fact.subject, "label", "")
                        if hasattr(fact, "subject")
                        else "",
                        relation_type=fact.relation.type.value
                        if hasattr(fact, "relation") and hasattr(fact.relation, "type")
                        else "",
                        object_label=getattr(fact.object, "label", "")
                        if hasattr(fact, "object")
                        else "",
                        confidence=getattr(fact, "confidence", 0.0),
                        temporal_mode="CURRENT"
                        if getattr(fact, "is_current", True)
                        else "HISTORY",
                        valid_from=str(fact.temporal.valid_from)
                        if hasattr(fact, "temporal")
                        else None,
                        valid_to=str(fact.temporal.valid_to)
                        if hasattr(fact, "temporal") and fact.temporal.valid_to
                        else None,
                        source=getattr(fact, "source", ""),
                    )
                )
            elif isinstance(fact, dict):
                converted.append(
                    RetrievedFact(
                        fact_id=fact.get("fact_id", ""),
                        subject_label=fact.get("subject_label", ""),
                        relation_type=fact.get("relation_type", ""),
                        object_label=fact.get("object_label", ""),
                        confidence=fact.get("confidence", 0.0),
                        temporal_mode=fact.get("temporal_mode", "CURRENT"),
                        valid_from=fact.get("valid_from"),
                        valid_to=fact.get("valid_to"),
                        source=fact.get("source", ""),
                    )
                )
        return converted

    def _convert_decisions(
        self,
        raw_decisions: List[Any],
        threshold: float,
    ) -> List[RouterDecision]:
        """Convert raw router decisions to structured format."""
        converted: List[RouterDecision] = []
        for i, decision in enumerate(raw_decisions):
            if isinstance(decision, dict):
                prob = decision.get("probability", decision.get("prob", 0.0))
                is_retrieval = decision.get("retrieval", prob >= threshold)
                n_facts = decision.get("num_facts", 0)
            elif isinstance(decision, (float, int)):
                prob = float(decision)
                is_retrieval = prob >= threshold
                n_facts = 0
            else:
                prob = float(getattr(decision, "probability", 0.0))
                is_retrieval = prob >= threshold
                n_facts = getattr(decision, "num_facts", 0)

            converted.append(
                RouterDecision(
                    step=i,
                    probability=prob,
                    decision="retrieval" if is_retrieval else "generation",
                    num_facts_retrieved=n_facts,
                )
            )
        return converted

    def warmup(self, warmup_text: str = "Biomedical warmup query.") -> None:
        """Run a warmup inference to pre-load model weights.

        Parameters
        ----------
        warmup_text : str
            Short text for warmup inference.
        """
        logger.info("Running warmup inference...")
        start = time.time()
        _ = self.generate(warmup_text, max_length=32)
        elapsed = (time.time() - start) * 1000
        logger.info("Warmup complete (%.1fms)", elapsed)

    def get_config_summary(self) -> Dict[str, Any]:
        """Return a summary of the pipeline configuration."""
        return {
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "router_threshold": self.router_threshold,
            "repetition_penalty": self.repetition_penalty,
            "has_faiss_index": self.faiss_index is not None,
            "has_kg_client": self.kg_client is not None,
        }
