"""
FRLM Configuration Loader.

Pydantic models that load, validate, and provide typed access to all
configuration parameters. Sensitive values (API keys, passwords, Neo4j
URLs and credentials) are read from ``config/secrets.properties`` and
override the values found in the YAML config file.

Usage:
    from config.config import load_config
    cfg = load_config()                          # loads config/default.yaml
    cfg = load_config("config/custom.yaml")      # loads a custom config
    cfg = load_config(overrides={"training.router.epochs": 20})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper to resolve project root
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent


def _resolve_path(p: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(p)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Secrets loader — reads key=value from .properties files
# ---------------------------------------------------------------------------
_SECRETS_CACHE: Optional[Dict[str, str]] = None


def _load_secrets(
    secrets_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """
    Load secrets from a Java-style ``.properties`` file.

    The file format is simple::

        # comment lines are ignored
        key=value
        key = value      # leading/trailing whitespace is stripped

    Parameters
    ----------
    secrets_path : str or Path, optional
        Explicit path to the properties file.  When *None* the loader
        looks for ``config/secrets.properties`` relative to the project
        root.

    Returns
    -------
    dict[str, str]
        Mapping of property keys to their string values.
        Returns an empty dict if the file does not exist.
    """
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None and secrets_path is None:
        return _SECRETS_CACHE

    if secrets_path is None:
        secrets_path = _CONFIG_DIR / "secrets.properties"
    else:
        secrets_path = Path(secrets_path)

    secrets: Dict[str, str] = {}

    if not secrets_path.exists():
        logger.debug("Secrets file not found: %s — skipping", secrets_path)
        _SECRETS_CACHE = secrets
        return secrets

    logger.info("Loading secrets from %s", secrets_path)
    with open(secrets_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                logger.warning(
                    "Ignoring malformed line %d in %s: %r",
                    lineno,
                    secrets_path,
                    line,
                )
                continue
            key, _, value = line.partition("=")
            secrets[key.strip()] = value.strip()

    _SECRETS_CACHE = secrets
    return secrets


def get_secret(key: str, default: str = "") -> str:
    """
    Retrieve a single secret by property key.

    Parameters
    ----------
    key : str
        The property key, e.g. ``"neo4j.password"`` or
        ``"anthropic.api_key"``.
    default : str
        Value to return when the key is not present.

    Returns
    -------
    str
    """
    return _load_secrets().get(key, default)


def reload_secrets(
    secrets_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Force-reload secrets from disk (clears cache)."""
    global _SECRETS_CACHE
    _SECRETS_CACHE = None
    return _load_secrets(secrets_path)


# ===========================================================================
# Sub-models — one per YAML section
# ===========================================================================


class ProjectConfig(BaseModel):
    """Project-level metadata."""

    name: str = "frlm"
    version: str = "0.1.0"
    description: str = ""
    seed: int = 42
    deterministic: bool = True


class PathsConfig(BaseModel):
    """All filesystem paths used by the project."""

    project_root: str = "."
    data_dir: str = "data"
    corpus_dir: str = "data/corpus"
    processed_dir: str = "data/processed"
    kg_dir: str = "data/kg"
    labels_dir: str = "data/labels"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    faiss_index_dir: str = "data/faiss_indices"
    export_dir: str = "exports"

    def resolve(self, field_name: str) -> Path:
        """Return an absolute Path for the given field."""
        return _resolve_path(getattr(self, field_name))

    def ensure_dirs(self) -> None:
        """Create all configured directories if they do not exist."""
        for field_name in type(self).model_fields:
            path = self.resolve(field_name)
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory: %s", path)


# ---- Model sub-configs ----------------------------------------------------


class BackboneConfig(BaseModel):
    """BioMedLM backbone configuration."""

    name: str = "stanford-crfm/BioMedLM"
    hidden_dim: int = 2560
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 50257
    max_seq_length: int = 1024
    dtype: str = "float16"
    gradient_checkpointing: bool = True
    freeze_backbone: bool = False
    freeze_layers: int = 0

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        allowed = {"float16", "float32", "bfloat16"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got '{v}'")
        return v


class RouterHeadConfig(BaseModel):
    """Router head configuration."""

    input_dim: int = 2560
    hidden_dim: int = 256
    output_dim: int = 1
    activation: str = "relu"
    dropout: float = 0.1
    threshold: float = 0.5

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v


class SemanticSubHeadConfig(BaseModel):
    """Semantic sub-head of the retrieval head."""

    input_dim: int = 2560
    output_dim: int = 768
    normalize: bool = True


class GranularitySubHeadConfig(BaseModel):
    """Granularity sub-head of the retrieval head."""

    input_dim: int = 2560
    num_levels: int = 4
    level_names: List[str] = Field(
        default=["atomic", "relation", "entity", "cluster"]
    )

    @field_validator("level_names")
    @classmethod
    def validate_level_names(cls, v: List[str], info: Any) -> List[str]:
        num_levels = info.data.get("num_levels", 4)
        if len(v) != num_levels:
            raise ValueError(
                f"level_names length ({len(v)}) must match num_levels ({num_levels})"
            )
        return v


class TemporalSubHeadConfig(BaseModel):
    """Temporal sub-head of the retrieval head."""

    input_dim: int = 2560
    num_modes: int = 3
    mode_names: List[str] = Field(
        default=["CURRENT", "AT_TIMESTAMP", "HISTORY"]
    )

    @field_validator("mode_names")
    @classmethod
    def validate_mode_names(cls, v: List[str], info: Any) -> List[str]:
        num_modes = info.data.get("num_modes", 3)
        if len(v) != num_modes:
            raise ValueError(
                f"mode_names length ({len(v)}) must match num_modes ({num_modes})"
            )
        return v


class RetrievalHeadConfig(BaseModel):
    """Retrieval head configuration (all three sub-heads)."""

    semantic: SemanticSubHeadConfig = Field(default_factory=SemanticSubHeadConfig)
    granularity: GranularitySubHeadConfig = Field(
        default_factory=GranularitySubHeadConfig
    )
    temporal: TemporalSubHeadConfig = Field(default_factory=TemporalSubHeadConfig)


class GenerationHeadConfig(BaseModel):
    """Generation head (standard LM head) configuration."""

    input_dim: int = 2560
    output_dim: int = 50257
    tie_weights: bool = True


class ModelConfig(BaseModel):
    """Full model configuration."""

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    router_head: RouterHeadConfig = Field(default_factory=RouterHeadConfig)
    retrieval_head: RetrievalHeadConfig = Field(default_factory=RetrievalHeadConfig)
    generation_head: GenerationHeadConfig = Field(default_factory=GenerationHeadConfig)


# ---- SapBERT ---------------------------------------------------------------


class SapBERTConfig(BaseModel):
    """Frozen SapBERT encoder configuration."""

    model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    embedding_dim: int = 768
    max_length: int = 64
    batch_size: int = 256
    device: str = "cuda"
    dtype: str = "float16"
    pool_strategy: str = "cls"

    @field_validator("pool_strategy")
    @classmethod
    def validate_pool_strategy(cls, v: str) -> str:
        allowed = {"cls", "mean", "max"}
        if v not in allowed:
            raise ValueError(f"pool_strategy must be one of {allowed}, got '{v}'")
        return v


# ---- Neo4j ------------------------------------------------------------------


class Neo4jSchemaConfig(BaseModel):
    """Neo4j schema labels and types."""

    fact_label: str = "Fact"
    entity_label: str = "Entity"
    relation_type: str = "HAS_FACT"
    version_chain_type: str = "SUPERSEDES"
    hash_algorithm: str = "sha256"
    hash_separator: str = "||"


class Neo4jBatchConfig(BaseModel):
    """Neo4j batch operation settings."""

    import_batch_size: int = 5000
    query_batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0


class Neo4jConfig(BaseModel):
    """Neo4j connection and schema configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "CHANGE_ME"
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    max_transaction_retry_time: int = 30
    encrypted: bool = False
    trust: str = "TRUST_ALL_CERTIFICATES"
    graph_schema: Neo4jSchemaConfig = Field(default_factory=Neo4jSchemaConfig)
    batch: Neo4jBatchConfig = Field(default_factory=Neo4jBatchConfig)

    @model_validator(mode="after")
    def override_from_secrets(self) -> "Neo4jConfig":
        secrets = _load_secrets()
        uri = secrets.get("neo4j.uri")
        if uri:
            self.uri = uri
            logger.info("Neo4j URI overridden from secrets.properties")
        username = secrets.get("neo4j.username")
        if username:
            self.username = username
            logger.info("Neo4j username overridden from secrets.properties")
        password = secrets.get("neo4j.password")
        if password:
            self.password = password
            logger.info("Neo4j password overridden from secrets.properties")
        database = secrets.get("neo4j.database")
        if database:
            self.database = database
            logger.info("Neo4j database overridden from secrets.properties")
        return self


# ---- FAISS ------------------------------------------------------------------


class SimilarityRangeConfig(BaseModel):
    """Cosine similarity range for hard negative mining."""

    min: float = 0.3
    max: float = 0.8


class HardNegativesConfig(BaseModel):
    """Hard negative mining configuration."""

    num_hard_negatives: int = 15
    num_random_negatives: int = 5
    mine_frequency: int = 1000
    similarity_range: SimilarityRangeConfig = Field(
        default_factory=SimilarityRangeConfig
    )


class HierarchicalIndexConfig(BaseModel):
    """Hierarchical index level mapping."""

    level_0: str = "atomic"
    level_1: str = "relation"
    level_2: str = "entity"
    level_3: str = "cluster"
    default_level: int = 0


class FAISSConfig(BaseModel):
    """FAISS vector index configuration."""

    index_type: str = "IVF4096,PQ64"
    metric: str = "L2"
    embedding_dim: int = 768
    nprobe: int = 64
    nlist: int = 4096
    pq_m: int = 64
    pq_nbits: int = 8
    use_gpu: bool = True
    gpu_id: int = 0
    train_sample_size: int = 500000
    search_k: int = 100
    use_precomputed_table: bool = True
    hierarchical: HierarchicalIndexConfig = Field(
        default_factory=HierarchicalIndexConfig
    )
    hard_negatives: HardNegativesConfig = Field(default_factory=HardNegativesConfig)

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        allowed = {"L2", "IP"}
        if v not in allowed:
            raise ValueError(f"metric must be one of {allowed}, got '{v}'")
        return v


# ---- Extraction -------------------------------------------------------------


class EntityExtractionConfig(BaseModel):
    """SciSpacy NER + UMLS linking configuration."""

    spacy_model: str = "en_core_sci_lg"
    linker: str = "umls"
    resolve_abbreviations: bool = True
    min_entity_length: int = 2
    max_entity_length: int = 100
    confidence_threshold: float = 0.7
    batch_size: int = 128
    n_process: int = 4


class RelationExtractionConfig(BaseModel):
    """Claude API relation extraction configuration."""

    model: str = "claude-sonnet-4-6"
    api_key: str = "CHANGE_ME"
    max_tokens: int = 4096
    temperature: float = 0.0
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_rpm: int = 50
    rate_limit_tpm: int = 100000
    system_prompt: str = ""

    @model_validator(mode="after")
    def override_api_key_from_secrets(self) -> "RelationExtractionConfig":
        api_key = get_secret("anthropic.api_key")
        if api_key:
            self.api_key = api_key
            logger.info(
                "Relation extraction API key overridden from secrets.properties"
            )
        return self


class CorpusConfig(BaseModel):
    """PMC corpus loading configuration."""

    source: str = "pmc_oa"
    format: str = "xml"
    max_documents: Optional[int] = None
    filter_journals: List[str] = Field(default_factory=list)
    min_year: int = 2000
    max_year: Optional[int] = None
    sections: List[str] = Field(
        default=["abstract", "introduction", "methods", "results", "discussion"]
    )
    chunk_size: int = 512
    chunk_overlap: int = 64


class ExtractionConfig(BaseModel):
    """All extraction pipeline configuration."""

    entity: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    relation: RelationExtractionConfig = Field(
        default_factory=RelationExtractionConfig
    )
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)


# ---- Labeling ---------------------------------------------------------------


class LabelValidationConfig(BaseModel):
    """Label quality validation thresholds."""

    min_retrieval_ratio: float = 0.15
    max_retrieval_ratio: float = 0.70
    min_spans_per_chunk: int = 1
    max_spans_per_chunk: int = 50


class LabelingConfig(BaseModel):
    """Claude API labeling configuration."""

    model: str = "claude-sonnet-4-6"
    api_key: str = "CHANGE_ME"
    max_tokens: int = 4096
    temperature: float = 0.0
    batch_size: int = 5
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_rpm: int = 50
    inter_annotator_samples: int = 200
    min_agreement_threshold: float = 0.8
    system_prompt: str = ""
    validation: LabelValidationConfig = Field(default_factory=LabelValidationConfig)

    # --- Cost-reduction options ---
    use_heuristic: bool = True        # heuristic pre-labeling for obvious cases
    api_batch_size: int = 50          # texts per API call (1 = legacy, 50 = recommended)

    @model_validator(mode="after")
    def override_api_key_from_secrets(self) -> "LabelingConfig":
        api_key = get_secret("anthropic.api_key")
        if api_key:
            self.api_key = api_key
            logger.info("Labeling API key overridden from secrets.properties")
        return self


# ---- Training ---------------------------------------------------------------


class RouterTrainingConfig(BaseModel):
    """Phase 1: Router pre-training parameters."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    label_smoothing: float = 0.05
    pos_weight: float = 1.5
    early_stopping_patience: int = 3
    early_stopping_metric: str = "f1"
    freeze_backbone: bool = True

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        allowed = {"linear", "cosine", "cosine_with_restarts"}
        if v not in allowed:
            raise ValueError(f"scheduler must be one of {allowed}, got '{v}'")
        return v


class RetrievalTrainingConfig(BaseModel):
    """Phase 2: Retrieval head training parameters."""

    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    contrastive_temperature: float = 0.07
    margin: float = 0.2
    early_stopping_patience: int = 5
    early_stopping_metric: str = "precision_at_1"
    freeze_backbone: bool = False
    freeze_router: bool = True

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        allowed = {"linear", "cosine", "cosine_with_restarts"}
        if v not in allowed:
            raise ValueError(f"scheduler must be one of {allowed}, got '{v}'")
        return v


class JointTrainingConfig(BaseModel):
    """Phase 3: Joint fine-tuning parameters."""

    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    scheduler: str = "cosine_with_restarts"
    num_cycles: int = 3
    early_stopping_patience: int = 5
    early_stopping_metric: str = "combined_loss"
    freeze_backbone: bool = False
    freeze_router: bool = False
    freeze_retrieval: bool = False

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        allowed = {"linear", "cosine", "cosine_with_restarts"}
        if v not in allowed:
            raise ValueError(f"scheduler must be one of {allowed}, got '{v}'")
        return v


class SplitsConfig(BaseModel):
    """Data split ratios."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1
    stratify_by: str = "retrieval_ratio"

    @model_validator(mode="after")
    def validate_split_sum(self) -> "SplitsConfig":
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Train/val/test splits must sum to 1.0, got {total:.6f}"
            )
        return self


class TrainingConfig(BaseModel):
    """Full training configuration."""

    output_dir: str = "checkpoints"
    seed: int = 42
    gpu_id: int = 0  # which CUDA device to use (0, 1, …); -1 = CPU
    fp16: bool = True
    bf16: bool = False
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 500
    save_every_n_steps: int = 1000
    max_checkpoints: int = 5
    router: RouterTrainingConfig = Field(default_factory=RouterTrainingConfig)
    retrieval: RetrievalTrainingConfig = Field(default_factory=RetrievalTrainingConfig)
    joint: JointTrainingConfig = Field(default_factory=JointTrainingConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)

    @model_validator(mode="after")
    def validate_precision(self) -> "TrainingConfig":
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16 simultaneously")
        return self


# ---- Loss -------------------------------------------------------------------


class LossConfig(BaseModel):
    """Combined loss function weights."""

    router_weight: float = 1.0
    retrieval_weight: float = 2.0
    generation_weight: float = 1.0
    contrastive_temperature: float = 0.07

    @field_validator("contrastive_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"contrastive_temperature must be > 0, got {v}")
        return v


# ---- DeepSpeed --------------------------------------------------------------


class DeepSpeedFP16Config(BaseModel):
    """DeepSpeed FP16 settings."""

    enabled: bool = True
    loss_scale: int = 0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1


class ZeroOffloadConfig(BaseModel):
    """DeepSpeed ZeRO offload settings."""

    device: str = "cpu"
    pin_memory: bool = True


class ZeroParamOffloadConfig(BaseModel):
    """DeepSpeed ZeRO parameter offload settings."""

    device: str = "none"


class ZeroOptimizationConfig(BaseModel):
    """DeepSpeed ZeRO optimization settings."""

    stage: int = 2
    offload_optimizer: ZeroOffloadConfig = Field(default_factory=ZeroOffloadConfig)
    offload_param: ZeroParamOffloadConfig = Field(
        default_factory=ZeroParamOffloadConfig
    )
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: float = 5.0e8
    allgather_bucket_size: float = 5.0e8


class DSOptimizerParamsConfig(BaseModel):
    """DeepSpeed optimizer parameters."""

    lr: Union[str, float] = "auto"
    betas: List[float] = Field(default=[0.9, 0.999])
    eps: float = 1.0e-8
    weight_decay: Union[str, float] = "auto"


class DSOptimizerConfig(BaseModel):
    """DeepSpeed optimizer settings."""

    type: str = "AdamW"
    params: DSOptimizerParamsConfig = Field(default_factory=DSOptimizerParamsConfig)


class DSSchedulerParamsConfig(BaseModel):
    """DeepSpeed scheduler parameters."""

    warmup_min_lr: Union[str, float] = 0
    warmup_max_lr: Union[str, float] = "auto"
    warmup_num_steps: Union[str, int] = "auto"
    total_num_steps: Union[str, int] = "auto"


class DSSchedulerConfig(BaseModel):
    """DeepSpeed scheduler settings."""

    type: str = "WarmupDecayLR"
    params: DSSchedulerParamsConfig = Field(default_factory=DSSchedulerParamsConfig)


class ActivationCheckpointingConfig(BaseModel):
    """DeepSpeed activation checkpointing settings."""

    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = False
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


class DeepSpeedInnerConfig(BaseModel):
    """DeepSpeed inner configuration (mirrors ds_config.json)."""

    train_batch_size: Union[str, int] = "auto"
    train_micro_batch_size_per_gpu: Union[str, int] = "auto"
    gradient_accumulation_steps: Union[str, int] = "auto"
    gradient_clipping: float = 1.0
    fp16: DeepSpeedFP16Config = Field(default_factory=DeepSpeedFP16Config)
    zero_optimization: ZeroOptimizationConfig = Field(
        default_factory=ZeroOptimizationConfig
    )
    optimizer: DSOptimizerConfig = Field(default_factory=DSOptimizerConfig)
    scheduler: DSSchedulerConfig = Field(default_factory=DSSchedulerConfig)
    activation_checkpointing: ActivationCheckpointingConfig = Field(
        default_factory=ActivationCheckpointingConfig
    )


class DeepSpeedConfig(BaseModel):
    """Top-level DeepSpeed configuration."""

    enabled: bool = True
    config: DeepSpeedInnerConfig = Field(default_factory=DeepSpeedInnerConfig)


# ---- WandB ------------------------------------------------------------------


class WandBConfig(BaseModel):
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "frlm"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = Field(default=["biomedical", "oncology", "retrieval-augmented"])
    log_model: bool = True
    log_frequency: int = 10
    watch_model: bool = False
    api_key: str = "CHANGE_ME"

    @model_validator(mode="after")
    def override_api_key_from_secrets(self) -> "WandBConfig":
        api_key = get_secret("wandb.api_key")
        if api_key:
            self.api_key = api_key
            logger.info("WandB API key overridden from secrets.properties")
        return self


# ---- Evaluation -------------------------------------------------------------


class RetrievalEvalConfig(BaseModel):
    """Retrieval evaluation settings."""

    k_values: List[int] = Field(default=[1, 5, 10, 20])
    temporal_accuracy: bool = True
    granularity_accuracy: bool = True
    num_eval_samples: int = 5000


class GenerationEvalConfig(BaseModel):
    """Generation evaluation settings."""

    max_length: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_eval_samples: int = 1000
    compute_perplexity: bool = True
    compute_bleu: bool = False
    compute_rouge: bool = False


class RouterEvalConfig(BaseModel):
    """Router evaluation settings."""

    threshold_sweep: List[float] = Field(default=[0.3, 0.4, 0.5, 0.6, 0.7])
    compute_confusion_matrix: bool = True
    compute_calibration: bool = True
    num_eval_samples: int = 5000


class EndToEndEvalConfig(BaseModel):
    """End-to-end evaluation settings."""

    num_eval_samples: int = 500
    compute_factual_accuracy: bool = True
    compute_temporal_consistency: bool = True


class EvaluationConfig(BaseModel):
    """Full evaluation configuration."""

    retrieval: RetrievalEvalConfig = Field(default_factory=RetrievalEvalConfig)
    generation: GenerationEvalConfig = Field(default_factory=GenerationEvalConfig)
    router: RouterEvalConfig = Field(default_factory=RouterEvalConfig)
    end_to_end: EndToEndEvalConfig = Field(default_factory=EndToEndEvalConfig)


# ---- Inference / Serving ----------------------------------------------------


class InferenceConfig(BaseModel):
    """Inference pipeline settings."""

    device: str = "cuda"
    dtype: str = "float16"
    max_length: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 4
    router_threshold: float = 0.3
    batch_size: int = 1


class ServingConfig(BaseModel):
    """FastAPI serving settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = Field(default=["*"])
    max_concurrent_requests: int = 16
    request_timeout: int = 60
    model_warmup: bool = True


# ---- Logging ----------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Python logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file: str = "logs/frlm.log"
    max_bytes: int = 10485760
    backup_count: int = 5
    console: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"level must be one of {allowed}, got '{v}'")
        return v_upper


# ===========================================================================
# Root configuration model
# ===========================================================================


class FRLMConfig(BaseModel):
    """
    Root FRLM configuration.

    Aggregates all sub-configurations into a single validated object.
    Sensitive fields are automatically overridden from
    ``config/secrets.properties``.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sapbert: SapBERTConfig = Field(default_factory=SapBERTConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    deepspeed: DeepSpeedConfig = Field(default_factory=DeepSpeedConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def cross_validate(self) -> "FRLMConfig":
        """Validate cross-field consistency."""
        hd = self.model.backbone.hidden_dim

        if self.model.router_head.input_dim != hd:
            raise ValueError(
                f"router_head.input_dim ({self.model.router_head.input_dim}) "
                f"must match backbone.hidden_dim ({hd})"
            )
        if self.model.retrieval_head.semantic.input_dim != hd:
            raise ValueError(
                f"retrieval_head.semantic.input_dim "
                f"({self.model.retrieval_head.semantic.input_dim}) "
                f"must match backbone.hidden_dim ({hd})"
            )
        if self.model.retrieval_head.granularity.input_dim != hd:
            raise ValueError(
                f"retrieval_head.granularity.input_dim "
                f"({self.model.retrieval_head.granularity.input_dim}) "
                f"must match backbone.hidden_dim ({hd})"
            )
        if self.model.retrieval_head.temporal.input_dim != hd:
            raise ValueError(
                f"retrieval_head.temporal.input_dim "
                f"({self.model.retrieval_head.temporal.input_dim}) "
                f"must match backbone.hidden_dim ({hd})"
            )
        if self.model.generation_head.input_dim != hd:
            raise ValueError(
                f"generation_head.input_dim ({self.model.generation_head.input_dim}) "
                f"must match backbone.hidden_dim ({hd})"
            )
        if self.model.generation_head.output_dim != self.model.backbone.vocab_size:
            raise ValueError(
                f"generation_head.output_dim ({self.model.generation_head.output_dim}) "
                f"must match backbone.vocab_size ({self.model.backbone.vocab_size})"
            )

        sapbert_dim = self.sapbert.embedding_dim
        if self.model.retrieval_head.semantic.output_dim != sapbert_dim:
            raise ValueError(
                f"retrieval_head.semantic.output_dim "
                f"({self.model.retrieval_head.semantic.output_dim}) "
                f"must match sapbert.embedding_dim ({sapbert_dim})"
            )
        if self.faiss.embedding_dim != sapbert_dim:
            raise ValueError(
                f"faiss.embedding_dim ({self.faiss.embedding_dim}) "
                f"must match sapbert.embedding_dim ({sapbert_dim})"
            )

        return self


# ===========================================================================
# Loading helpers
# ===========================================================================


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *override* into *base*. Values in *override* take
    precedence. Neither input dict is mutated.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dot_overrides(
    data: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply dot-notation overrides like ``{"training.router.epochs": 20}``.
    """
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        target = data
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    return data


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> FRLMConfig:
    """
    Load and validate the FRLM configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config file. Defaults to ``config/default.yaml``
        relative to the project root.
    overrides : dict, optional
        Dot-notation overrides applied after loading the YAML file.
        Example: ``{"training.router.epochs": 20}``

    Returns
    -------
    FRLMConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    pydantic.ValidationError
        If the configuration fails validation.
    """
    if config_path is None:
        config_path = _CONFIG_DIR / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info("Loading configuration from %s", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data: Dict[str, Any] = yaml.safe_load(f) or {}

    if overrides:
        raw_data = _apply_dot_overrides(raw_data, overrides)
        logger.info("Applied %d configuration overrides", len(overrides))

    config = FRLMConfig(**raw_data)
    logger.info(
        "Configuration loaded and validated successfully (project=%s, version=%s)",
        config.project.name,
        config.project.version,
    )
    return config


def load_and_merge_configs(
    base_path: Union[str, Path],
    override_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> FRLMConfig:
    """
    Load a base config and merge an override config on top.

    Useful for experiment-specific configs that only specify deltas
    from the default configuration.

    Parameters
    ----------
    base_path : str or Path
        Path to the base YAML config.
    override_path : str or Path
        Path to the override YAML config.
    overrides : dict, optional
        Additional dot-notation overrides applied last.

    Returns
    -------
    FRLMConfig
        Fully validated merged configuration.
    """
    base_path = Path(base_path)
    override_path = Path(override_path)

    for p in [base_path, override_path]:
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(base_path, "r", encoding="utf-8") as f:
        base_data: Dict[str, Any] = yaml.safe_load(f) or {}

    with open(override_path, "r", encoding="utf-8") as f:
        override_data: Dict[str, Any] = yaml.safe_load(f) or {}

    merged = _deep_merge(base_data, override_data)

    if overrides:
        merged = _apply_dot_overrides(merged, overrides)

    config = FRLMConfig(**merged)
    logger.info(
        "Merged configuration loaded: base=%s, override=%s", base_path, override_path
    )
    return config


def setup_logging(config: FRLMConfig) -> None:
    """
    Configure the Python logging system from the config.

    Parameters
    ----------
    config : FRLMConfig
        Validated FRLM configuration.
    """
    from logging.handlers import RotatingFileHandler

    log_cfg = config.logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_cfg.level))

    formatter = logging.Formatter(fmt=log_cfg.format, datefmt=log_cfg.date_format)

    # Console handler
    if log_cfg.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    log_file = _resolve_path(log_cfg.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=log_cfg.max_bytes,
        backupCount=log_cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging configured: level=%s, file=%s", log_cfg.level, log_file)