"""
FRLM — Factual Retrieval Language Model (full composite model).

Combines:

- :class:`BioMedLMBackbone` — decoder-only transformer
- :class:`RouterHead` — binary retrieval / generation classifier
- :class:`RetrievalHead` — structured query signature for KG lookup
- :class:`GenerationHead` — standard LM head for next-token prediction
- :class:`FRLMCombinedLoss` — weighted multi-task loss

Public API
----------
- ``forward(...)`` — training forward pass returning :class:`FRLMOutput`
- ``generate(input_ids, ...)`` — autoregressive generation with interleaved
  retrieval and generation
- ``save_pretrained(path)`` / ``from_pretrained(path)`` — persistence
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.model.backbone import BackboneOutput, BioMedLMBackbone
from src.model.generation_head import GenerationHead
from src.model.losses import FRLMCombinedLoss
from src.model.retrieval_head import QuerySignature, RetrievalHead
from src.model.router_head import RouterHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class FRLMOutput:
    """Container returned by :meth:`FRLMModel.forward`.

    All fields are optional so that the model can be used in different
    modes (training with losses, inference without labels, etc.).

    Attributes
    ----------
    last_hidden_state : Tensor
        Final-layer backbone hidden states ``(batch, seq_len, hidden_dim)``.
    router_logits : Tensor
        Raw router logits ``(batch, seq_len, 1)``.
    router_probs : Tensor
        Router probabilities ``(batch, seq_len, 1)``.
    router_mask : Tensor
        Boolean mask ``(batch, seq_len)`` — True = retrieval.
    query_signature : QuerySignature or None
        Retrieval head output (only for retrieval positions).
    gen_logits : Tensor or None
        Generation logits ``(batch, seq_len, vocab_size)``.
    total_loss : Tensor or None
        Scalar combined loss (only when labels are provided).
    loss_dict : dict[str, Tensor] or None
        Individual loss components for logging.
    """

    last_hidden_state: Tensor
    router_logits: Tensor
    router_probs: Tensor
    router_mask: Tensor
    query_signature: Optional[QuerySignature] = None
    gen_logits: Optional[Tensor] = None
    total_loss: Optional[Tensor] = None
    loss_dict: Optional[Dict[str, Tensor]] = None


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class FRLMModel(nn.Module):
    """Factual Retrieval Language Model.

    Parameters
    ----------
    backbone : BioMedLMBackbone
        Pre-loaded backbone.
    router : RouterHead
        Router classification head.
    retrieval_head : RetrievalHead
        Structured query head.
    generation_head : GenerationHead
        Standard LM head.
    loss_fn : FRLMCombinedLoss, optional
        Combined loss (omit for inference-only).
    router_threshold : float
        Decision boundary for the router.
    """

    def __init__(
        self,
        backbone: BioMedLMBackbone,
        router: RouterHead,
        retrieval_head: RetrievalHead,
        generation_head: GenerationHead,
        loss_fn: Optional[FRLMCombinedLoss] = None,
        router_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.router = router
        self.retrieval_head = retrieval_head
        self.generation_head = generation_head
        self.loss_fn = loss_fn
        self._router_threshold = router_threshold

        logger.info("FRLMModel assembled (threshold=%.2f)", router_threshold)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self.backbone.get_hidden_dim()

    @property
    def router_threshold(self) -> float:
        return self._router_threshold

    @router_threshold.setter
    def router_threshold(self, value: float) -> None:
        self._router_threshold = value
        self.router.threshold = value

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        router_labels: Optional[Tensor] = None,
        fact_embeddings: Optional[Tensor] = None,
        negative_embeddings: Optional[Tensor] = None,
        token_labels: Optional[Tensor] = None,
    ) -> FRLMOutput:
        """Training / evaluation forward pass.

        Parameters
        ----------
        input_ids : Tensor
            Token ids ``(batch, seq_len)``.
        attention_mask : Tensor, optional
            Padding mask ``(batch, seq_len)``.
        router_labels : Tensor, optional
            Binary labels ``(batch, seq_len)`` — 1 = retrieval, 0 = generation.
        fact_embeddings : Tensor, optional
            Positive fact SapBERT embeddings ``(batch, seq_len, emb_dim)``
            aligned per-position (non-retrieval positions ignored).
        negative_embeddings : Tensor, optional
            Hard-negative embeddings ``(batch, seq_len, num_neg, emb_dim)``.
        token_labels : Tensor, optional
            Next-token labels ``(batch, seq_len)`` for generation loss.

        Returns
        -------
        FRLMOutput
            Composite output with all tensors and optional losses.
        """
        # 1. Backbone
        backbone_out: BackboneOutput = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = backbone_out.last_hidden_state  # (batch, seq, hid)

        # 2. Router
        router_logits = self.router.forward(hidden)          # (batch, seq, 1)
        router_probs = torch.sigmoid(router_logits)          # (batch, seq, 1)
        router_mask = (router_probs.squeeze(-1) > self._router_threshold)  # (batch, seq)

        # 3. Retrieval head (applied to all positions; loss is filtered later)
        query_sig: QuerySignature = self.retrieval_head.forward(hidden)

        # 4. Generation head
        gen_logits = self.generation_head.forward(hidden)  # (batch, seq, vocab)

        # 5. Loss computation (if labels are provided)
        total_loss: Optional[Tensor] = None
        loss_dict: Optional[Dict[str, Tensor]] = None

        if router_labels is not None and self.loss_fn is not None:
            # Gather retrieval-position embeddings for contrastive loss
            query_emb: Optional[Tensor] = None
            pos_emb: Optional[Tensor] = None
            neg_embs: Optional[Tensor] = None

            if fact_embeddings is not None:
                # Collect only retrieval positions (where router_labels == 1)
                ret_mask = router_labels.bool()  # (batch, seq)

                if ret_mask.any():
                    # Flatten retrieval positions
                    query_emb = query_sig.semantic_embedding[ret_mask]  # (N, emb_dim)
                    pos_emb = fact_embeddings[ret_mask]                  # (N, emb_dim)

                    if negative_embeddings is not None:
                        neg_embs = negative_embeddings[ret_mask]  # (N, num_neg, emb_dim)
                    else:
                        # Provide a dummy if no negatives supplied
                        neg_embs = torch.zeros(
                            query_emb.size(0), 1, query_emb.size(1),
                            device=query_emb.device, dtype=query_emb.dtype,
                        )

            total_loss, loss_dict = self.loss_fn(
                router_logits=router_logits.squeeze(-1),
                router_labels=router_labels,
                query_emb=query_emb,
                positive_emb=pos_emb,
                negative_embs=neg_embs,
                gen_logits=gen_logits,
                gen_labels=token_labels,
                router_mask=router_labels.float(),
                attention_mask=attention_mask,
            )

        return FRLMOutput(
            last_hidden_state=hidden,
            router_logits=router_logits,
            router_probs=router_probs,
            router_mask=router_mask,
            query_signature=query_sig,
            gen_logits=gen_logits,
            total_loss=total_loss,
            loss_dict=loss_dict,
        )

    # ------------------------------------------------------------------
    # Autoregressive generation with interleaved retrieval
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Static helpers for sampling filters
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(
        logits: Tensor,
        generated_ids: Tensor,
        penalty: float,
    ) -> Tensor:
        """Apply multiplicative repetition penalty (Keskar et al., 2019).

        For every token that already appears in *generated_ids*, divide
        its logit by *penalty* if positive, or multiply by *penalty* if
        negative.  ``penalty = 1.0`` is a no-op.

        Parameters
        ----------
        logits : Tensor
            Shape ``(1, vocab_size)``.
        generated_ids : Tensor
            Shape ``(1, seq_len)`` — token ids generated so far.
        penalty : float
            Repetition penalty factor (> 1.0 to penalise repetition).

        Returns
        -------
        Tensor
            Adjusted logits.
        """
        if penalty == 1.0:
            return logits
        score = torch.gather(logits, 1, generated_ids)
        # If score < 0 then multiply by penalty (make more negative)
        # If score > 0 then divide by penalty (make less positive)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, generated_ids, score)
        return logits

    @staticmethod
    def _has_repeated_ngram(
        token_ids: List[int],
        ngram_size: int,
    ) -> bool:
        """Return True if the last *ngram_size* tokens form an n-gram
        that already appeared earlier in the sequence."""
        if len(token_ids) < ngram_size * 2:
            return False
        last_ngram = tuple(token_ids[-ngram_size:])
        for i in range(len(token_ids) - ngram_size * 2 + 1):
            if tuple(token_ids[i : i + ngram_size]) == last_ngram:
                return True
        return False

    @staticmethod
    def _block_repeated_ngrams(
        logits: Tensor,
        generated_ids: List[int],
        ngram_size: int,
    ) -> Tensor:
        """Set logits to ``-inf`` for any token that would create a
        repeated *ngram_size*-gram.

        Parameters
        ----------
        logits : Tensor
            Shape ``(1, vocab_size)``.
        generated_ids : list of int
            Tokens generated so far.
        ngram_size : int
            N-gram order to block (e.g. 3 for tri-gram blocking).

        Returns
        -------
        Tensor
            Adjusted logits.
        """
        if len(generated_ids) < ngram_size:
            return logits
        # Collect all n-grams so far (except the incomplete last one)
        ngrams: dict[tuple[int, ...], list[int]] = {}
        for i in range(len(generated_ids) - ngram_size + 1):
            prefix = tuple(generated_ids[i : i + ngram_size - 1])
            continuation = generated_ids[i + ngram_size - 1]
            ngrams.setdefault(prefix, []).append(continuation)
        # The current prefix is the last (ngram_size - 1) tokens
        current_prefix = tuple(generated_ids[-(ngram_size - 1) :])
        banned_tokens = ngrams.get(current_prefix, [])
        if banned_tokens:
            logits[0, banned_tokens] = float("-inf")
        return logits

    # ------------------------------------------------------------------
    # Autoregressive generation with interleaved retrieval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        faiss_index: Any = None,
        kg_client: Any = None,
        tokenizer: Any = None,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        retrieval_top_k: int = 10,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 4,
    ) -> Dict[str, Any]:
        """Autoregressive generation with interleaved retrieval.

        At each step the router decides whether to **retrieve** a fact
        (and append its text representation) or **generate** the next
        token from the LM head.

        Parameters
        ----------
        input_ids : Tensor
            Prompt token ids ``(1, prompt_len)``.
        attention_mask : Tensor, optional
            Padding mask for the prompt.
        faiss_index : FAISSFactIndex or HierarchicalIndex, optional
            Vector index for retrieval.
        kg_client : Neo4jClient, optional
            KG client for temporal resolution.
        tokenizer : PreTrainedTokenizer, optional
            Tokenizer for encoding retrieved fact text into token ids.
            When provided, retrieved facts are injected into the token
            stream; otherwise they are only recorded.
        max_length : int
            Maximum output sequence length.
        temperature : float
            Softmax temperature for generation sampling.
        top_k : int
            Top-k filtering for generation.
        top_p : float
            Nucleus (top-p) filtering.
        retrieval_top_k : int
            Number of facts to retrieve per retrieval step.
        repetition_penalty : float
            Multiplicative penalty for tokens already generated
            (1.0 = no penalty, > 1.0 penalises repetition).
        no_repeat_ngram_size : int
            Block generation of any n-gram that already appeared in the
            output (0 = disabled).

        Returns
        -------
        dict
            ``"token_ids"`` — generated sequence (LongTensor),
            ``"retrieved_facts"`` — list of retrieved facts per step,
            ``"router_decisions"`` — list of per-step decisions.
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()
        retrieved_facts: List[Any] = []
        router_decisions: List[str] = []

        if attention_mask is None:
            attention_mask = torch.ones_like(generated)

        # Max context window for positional embeddings (GPT-2 family = 1024)
        max_ctx = getattr(self.backbone.transformer.config, "n_positions", 1024)
        prompt_len = input_ids.size(1)
        gen_steps = 0
        retrieval_cooldown = 0  # skip retrieval for N steps after a retrieval
        consecutive_repeat_count = 0  # track consecutive repeated tokens
        # Track model-generated tokens separately (excludes injected fact
        # tokens) so n-gram blocking is not disrupted by fact injections.
        gen_only_tokens: List[int] = []

        while gen_steps < max_length - prompt_len:
            # Sliding window: only feed the last max_ctx tokens to the backbone
            if generated.size(1) > max_ctx:
                ctx_ids = generated[:, -max_ctx:]
                ctx_mask = attention_mask[:, -max_ctx:]
            else:
                ctx_ids = generated
                ctx_mask = attention_mask

            # Forward
            backbone_out = self.backbone(ctx_ids, ctx_mask)
            hidden = backbone_out.last_hidden_state

            # Last position
            last_hidden = hidden[:, -1:, :]  # (1, 1, hid)

            # Router decision (suppress retrieval during cooldown)
            router_prob = torch.sigmoid(self.router.forward(last_hidden))  # (1, 1, 1)
            is_retrieval = (
                router_prob.item() > self._router_threshold
                and retrieval_cooldown <= 0
            )

            if is_retrieval and faiss_index is not None:
                router_decisions.append("retrieval")
                retrieval_cooldown = 3  # generate at least 3 tokens before next retrieval
                query_sig = self.retrieval_head.forward(last_hidden)
                facts = self.retrieval_head.resolve(
                    query_sig, faiss_index, kg_client, top_k=retrieval_top_k,
                )
                retrieved_facts.append(facts)

                # Tokenise retrieved fact text and inject into the sequence
                if facts and tokenizer is not None:
                    # Build a fact string: "subject relation object"
                    fact = facts[0]  # use top-1 retrieved fact
                    fact_text = ""
                    if isinstance(fact, dict):
                        fact_text = " ".join([
                            str(fact.get("subject", "")),
                            str(fact.get("relation", "")),
                            str(fact.get("object", "")),
                        ]).strip()
                    elif hasattr(fact, "subject") and hasattr(fact, "object"):
                        subj = getattr(fact.subject, "label", str(fact.subject))
                        rel = getattr(fact.relation, "type", str(fact.relation))
                        obj = getattr(fact.object, "label", str(fact.object))
                        fact_text = f"{subj} {rel} {obj}".strip()

                    # Only inject if we have real text (not just a hash ID)
                    if fact_text and len(fact_text) < 200:
                        fact_tokens = tokenizer.encode(
                            fact_text, add_special_tokens=False
                        )
                        fact_ids = torch.tensor(
                            [fact_tokens], dtype=torch.long, device=device,
                        )
                        generated = torch.cat([generated, fact_ids], dim=1)
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones(
                                    1, len(fact_tokens),
                                    device=device,
                                    dtype=attention_mask.dtype,
                                ),
                            ],
                            dim=1,
                        )
            else:
                router_decisions.append("generation")
                retrieval_cooldown = max(0, retrieval_cooldown - 1)
                logits = self.generation_head.forward(last_hidden)  # (1, 1, vocab)
                logits = logits[:, -1, :] / temperature  # (1, vocab)

                # --- Repetition penalty ---
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, generated, repetition_penalty,
                    )

                # --- N-gram blocking (uses model-generated tokens only) ---
                if no_repeat_ngram_size > 0:
                    logits = self._block_repeated_ngrams(
                        logits, gen_only_tokens, no_repeat_ngram_size,
                    )

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cum_probs > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    indices_to_remove = sorted_idx[remove]
                    logits[:, indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                generated = torch.cat([generated, next_token], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                    dim=1,
                )

                tok = next_token.item()
                gen_only_tokens.append(tok)

                # --- Early stop on degenerate repetition ---
                # 1) Same token emitted 6+ times in a row
                if generated.size(1) > prompt_len + 1:
                    prev_tok = generated[0, -2].item()
                    consecutive_repeat_count = (
                        consecutive_repeat_count + 1 if tok == prev_tok else 0
                    )
                    if consecutive_repeat_count >= 5:
                        logger.warning(
                            "Aborting generation: token %d repeated %d times",
                            tok,
                            consecutive_repeat_count + 1,
                        )
                        break

                # 2) Short-pattern loop detector: if a pattern of length
                #    2-6 tokens has been repeated 4+ times contiguously,
                #    the model is stuck in a degenerate loop.
                if len(gen_only_tokens) >= 12:
                    _abort = False
                    for pat_len in range(2, 7):
                        reps_needed = 4
                        window = reps_needed * pat_len
                        if len(gen_only_tokens) < window:
                            continue
                        tail = gen_only_tokens[-window:]
                        pattern = tail[:pat_len]
                        if all(
                            tail[j] == pattern[j % pat_len]
                            for j in range(window)
                        ):
                            logger.warning(
                                "Aborting generation: %d-token pattern "
                                "repeated %d times",
                                pat_len,
                                reps_needed,
                            )
                            _abort = True
                            break
                    if _abort:
                        break

                # Stop on EOS (token id 50256 for GPT-2 family)
                if tok == 50256:
                    break

            gen_steps += 1

        return {
            "token_ids": generated,
            "retrieved_facts": retrieved_facts,
            "router_decisions": router_decisions,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """Save all model components to *path*.

        Creates the directory structure::

            path/
                backbone/        — HuggingFace model files
                router.pt        — router head state dict
                retrieval.pt     — retrieval head state dict
                generation.pt    — generation head state dict
                loss.pt          — loss function state dict
                config.json      — model metadata

        Parameters
        ----------
        path : str or Path
            Root directory for the saved model.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Backbone (HuggingFace save)
        backbone_dir = save_dir / "backbone"
        backbone_dir.mkdir(exist_ok=True)
        self.backbone.transformer.save_pretrained(str(backbone_dir))

        # Heads
        torch.save(self.router.state_dict(), save_dir / "router.pt")
        torch.save(self.retrieval_head.state_dict(), save_dir / "retrieval.pt")
        torch.save(self.generation_head.state_dict(), save_dir / "generation.pt")

        # Loss (if present)
        if self.loss_fn is not None:
            torch.save(self.loss_fn.state_dict(), save_dir / "loss.pt")

        # Metadata
        meta = {
            "hidden_dim": self.hidden_dim,
            "router_threshold": self._router_threshold,
            "router_intermediate_dim": self.router._intermediate_dim,
            "embedding_dim": self.retrieval_head.embedding_dim,
            "num_granularity_levels": self.retrieval_head.num_granularity_levels,
            "num_temporal_modes": self.retrieval_head.num_temporal_modes,
            "vocab_size": self.backbone.vocab_size,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Model saved to %s", save_dir)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        load_loss: bool = False,
        device: str = "cpu",
    ) -> "FRLMModel":
        """Load a previously saved model.

        Parameters
        ----------
        path : str or Path
            Root directory that was passed to :meth:`save_pretrained`.
        load_loss : bool
            Whether to load the loss function state dict.
        device : str
            Device to map tensors to.

        Returns
        -------
        FRLMModel
        """
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")

        # Metadata
        with open(load_dir / "config.json", "r") as f:
            meta = json.load(f)

        hidden_dim = meta["hidden_dim"]
        vocab_size = meta["vocab_size"]
        emb_dim = meta["embedding_dim"]
        n_gran = meta["num_granularity_levels"]
        n_temp = meta["num_temporal_modes"]
        threshold = meta.get("router_threshold", 0.5)
        router_intermediate = meta.get("router_intermediate_dim", 256)

        # Backbone
        backbone = BioMedLMBackbone(
            model_name=str(load_dir / "backbone"),
            hidden_dim=hidden_dim,
            freeze_backbone=False,
            gradient_checkpointing=False,
        )

        # Heads
        router = RouterHead(hidden_dim=hidden_dim, intermediate_dim=router_intermediate)
        router.load_state_dict(
            torch.load(load_dir / "router.pt", map_location=device)
        )

        retrieval_head = RetrievalHead(
            hidden_dim=hidden_dim,
            embedding_dim=emb_dim,
            num_granularity_levels=n_gran,
            num_temporal_modes=n_temp,
        )
        retrieval_head.load_state_dict(
            torch.load(load_dir / "retrieval.pt", map_location=device)
        )

        generation_head = GenerationHead(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )
        generation_head.load_state_dict(
            torch.load(load_dir / "generation.pt", map_location=device)
        )

        # Loss
        loss_fn = None
        if load_loss and (load_dir / "loss.pt").exists():
            loss_fn = FRLMCombinedLoss()
            loss_fn.load_state_dict(
                torch.load(load_dir / "loss.pt", map_location=device)
            )

        model = cls(
            backbone=backbone,
            router=router,
            retrieval_head=retrieval_head,
            generation_head=generation_head,
            loss_fn=loss_fn,
            router_threshold=threshold,
        )
        logger.info("Model loaded from %s", load_dir)
        return model

    # ------------------------------------------------------------------
    # Factory (from FRLM config)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "FRLMConfig") -> "FRLMModel":  # type: ignore[name-defined]  # noqa: F821
        """Construct from a :class:`config.config.FRLMConfig` instance.

        Builds all components using their respective ``from_config``
        class methods and wires them together.

        Parameters
        ----------
        config : FRLMConfig
            Root FRLM configuration.
        """
        backbone = BioMedLMBackbone.from_config(config.model.backbone)
        router = RouterHead.from_config(config.model.router_head)
        retrieval_head = RetrievalHead.from_config(config.model.retrieval_head)
        generation_head = GenerationHead.from_config(
            config.model.generation_head, backbone=backbone,
        )
        loss_fn = FRLMCombinedLoss.from_config(config.loss, config.training)

        return cls(
            backbone=backbone,
            router=router,
            retrieval_head=retrieval_head,
            generation_head=generation_head,
            loss_fn=loss_fn,
            router_threshold=config.model.router_head.threshold,
        )
