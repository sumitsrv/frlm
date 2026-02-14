#!/usr/bin/env python3
"""
estimate_costs.py — Estimate resource costs for the FRLM pipeline.

Produces a detailed cost breakdown covering:
  • Claude API costs   (relation extraction & router labeling)
  • Compute costs      (GPU hours for training phases)
  • Storage costs      (Neo4j, FAISS index, checkpoints, logs)

Pipeline position: Utility script (not a numbered step)
Reads from:  config/default.yaml
Writes to:   stdout (human-readable) or JSON file (--json)
Config used: config.extraction, config.labeling, config.training, config.paths

Usage
-----
    python scripts/estimate_costs.py --config config/default.yaml
    python scripts/estimate_costs.py --config config/default.yaml --json costs.json
    python scripts/estimate_costs.py --config config/default.yaml --max-documents 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, load_config, setup_logging

logger = logging.getLogger(__name__)

# ===========================================================================
# Pricing constants (as of 2025-01)
# ===========================================================================

# Claude Sonnet pricing (per token)
CLAUDE_SONNET_INPUT_PRICE_PER_M = 3.00   # $/1M input tokens
CLAUDE_SONNET_OUTPUT_PRICE_PER_M = 15.00  # $/1M output tokens

# GPU hourly rates (on-demand, approximate)
GPU_HOURLY_RATES: Dict[str, float] = {
    "A100-80GB": 3.50,       # cloud A100 80 GB
    "A100-40GB": 2.50,       # cloud A100 40 GB
    "H100": 5.00,            # cloud H100
    "A10G": 1.50,            # cloud A10G
    "V100": 1.20,            # cloud V100
    "T4": 0.50,              # budget option
    "local": 0.00,           # local GPU (electricity only)
}

# Storage pricing (rough estimates)
STORAGE_COST_PER_GB_MONTH = 0.10  # generic SSD block storage


# ===========================================================================
# Data classes for structured cost estimates
# ===========================================================================


@dataclass
class ClaudeAPICost:
    """Cost estimate for a single Claude API step."""

    step_name: str
    model: str
    total_chunks: int
    input_tokens_per_chunk: int
    output_tokens_per_chunk: int
    total_input_tokens: int
    total_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    rate_limit_rpm: int
    estimated_minutes: float


@dataclass
class TrainingPhaseCost:
    """Cost estimate for a single training phase."""

    phase_name: str
    epochs: int
    batch_size: int
    estimated_batches_per_epoch: int
    seconds_per_batch: float
    total_gpu_hours: float
    gpu_type: str
    cost_usd: float


@dataclass
class StorageEstimate:
    """Estimated storage for a single component."""

    component: str
    estimated_gb: float
    monthly_cost_usd: float


@dataclass
class CostReport:
    """Full cost report aggregating all categories."""

    api_costs: List[ClaudeAPICost] = field(default_factory=list)
    training_costs: List[TrainingPhaseCost] = field(default_factory=list)
    storage_estimates: List[StorageEstimate] = field(default_factory=list)
    total_api_cost_usd: float = 0.0
    total_training_cost_usd: float = 0.0
    total_storage_monthly_usd: float = 0.0
    grand_total_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_costs": [asdict(c) for c in self.api_costs],
            "training_costs": [asdict(c) for c in self.training_costs],
            "storage_estimates": [asdict(s) for s in self.storage_estimates],
            "total_api_cost_usd": round(self.total_api_cost_usd, 2),
            "total_training_cost_usd": round(self.total_training_cost_usd, 2),
            "total_storage_monthly_usd": round(self.total_storage_monthly_usd, 2),
            "grand_total_usd": round(self.grand_total_usd, 2),
        }


# ===========================================================================
# Estimation functions
# ===========================================================================


def estimate_corpus_chunks(cfg: FRLMConfig, max_documents: Optional[int] = None) -> int:
    """Estimate the total number of text chunks the corpus will produce.

    Heuristic: a typical PMC OA paper has ~4 000 tokens of extractable text.
    With chunk_size = 512 and overlap, that yields ~12 chunks per document.
    """
    corpus_cfg = cfg.extraction.corpus
    n_docs = max_documents if max_documents is not None else (corpus_cfg.max_documents or 1000)
    chunk_size = corpus_cfg.chunk_size
    chunk_overlap = corpus_cfg.chunk_overlap

    avg_doc_tokens = 4000
    effective_chunk = chunk_size - chunk_overlap
    chunks_per_doc = max(1, math.ceil(avg_doc_tokens / effective_chunk))
    total_chunks = n_docs * chunks_per_doc

    logger.info(
        "Corpus estimate: %d docs × %d chunks/doc = %d chunks",
        n_docs, chunks_per_doc, total_chunks,
    )
    return total_chunks


def estimate_api_costs(cfg: FRLMConfig, total_chunks: int) -> List[ClaudeAPICost]:
    """Estimate Claude API costs for relation extraction and router labeling."""
    costs: List[ClaudeAPICost] = []

    # --- Step 3: Relation extraction ---
    rel_cfg = cfg.extraction.relation
    # Entity chunks → relation calls. Not every chunk goes to Claude;
    # only chunks with entities. Assume ~80% hit rate.
    rel_chunks = int(total_chunks * 0.80)
    rel_input_per = 2000   # system prompt + entity context
    rel_output_per = 500   # structured relation JSON
    rel_total_in = rel_chunks * rel_input_per
    rel_total_out = rel_chunks * rel_output_per
    rel_in_cost = rel_total_in * CLAUDE_SONNET_INPUT_PRICE_PER_M / 1_000_000
    rel_out_cost = rel_total_out * CLAUDE_SONNET_OUTPUT_PRICE_PER_M / 1_000_000
    rel_rpm = rel_cfg.rate_limit_rpm
    rel_minutes = math.ceil(rel_chunks / rel_rpm) if rel_rpm > 0 else 0

    costs.append(ClaudeAPICost(
        step_name="Relation extraction (Step 3)",
        model=rel_cfg.model,
        total_chunks=rel_chunks,
        input_tokens_per_chunk=rel_input_per,
        output_tokens_per_chunk=rel_output_per,
        total_input_tokens=rel_total_in,
        total_output_tokens=rel_total_out,
        input_cost_usd=round(rel_in_cost, 2),
        output_cost_usd=round(rel_out_cost, 2),
        total_cost_usd=round(rel_in_cost + rel_out_cost, 2),
        rate_limit_rpm=rel_rpm,
        estimated_minutes=rel_minutes,
    ))

    # --- Step 6: Router labeling ---
    lab_cfg = cfg.labeling
    lab_input_per = 1800   # system prompt + text chunk
    lab_output_per = 450   # labeling JSON
    lab_total_in = total_chunks * lab_input_per
    lab_total_out = total_chunks * lab_output_per
    lab_in_cost = lab_total_in * CLAUDE_SONNET_INPUT_PRICE_PER_M / 1_000_000
    lab_out_cost = lab_total_out * CLAUDE_SONNET_OUTPUT_PRICE_PER_M / 1_000_000
    lab_rpm = lab_cfg.rate_limit_rpm
    lab_minutes = math.ceil(total_chunks / lab_rpm) if lab_rpm > 0 else 0

    costs.append(ClaudeAPICost(
        step_name="Router labeling (Step 6)",
        model=lab_cfg.model,
        total_chunks=total_chunks,
        input_tokens_per_chunk=lab_input_per,
        output_tokens_per_chunk=lab_output_per,
        total_input_tokens=lab_total_in,
        total_output_tokens=lab_total_out,
        input_cost_usd=round(lab_in_cost, 2),
        output_cost_usd=round(lab_out_cost, 2),
        total_cost_usd=round(lab_in_cost + lab_out_cost, 2),
        rate_limit_rpm=lab_rpm,
        estimated_minutes=lab_minutes,
    ))

    return costs


def estimate_training_costs(
    cfg: FRLMConfig,
    total_chunks: int,
    gpu_type: str = "A100-80GB",
) -> List[TrainingPhaseCost]:
    """Estimate GPU training costs for all three phases.

    Heuristics:
      - Training samples ≈ total_chunks (label files + entity pairs)
      - Seconds per batch scale with batch size and model params.
      - Phase 3 (joint) is ~2× slower per batch due to larger graph.
    """
    gpu_rate = GPU_HOURLY_RATES.get(gpu_type, GPU_HOURLY_RATES["A100-80GB"])
    costs: List[TrainingPhaseCost] = []
    training_samples = total_chunks

    phases = [
        (
            "Phase 1 — Router head",
            cfg.training.router.epochs,
            cfg.training.router.batch_size,
            0.35,  # seconds per batch
        ),
        (
            "Phase 2 — Retrieval head",
            cfg.training.retrieval.epochs,
            cfg.training.retrieval.batch_size,
            0.55,
        ),
        (
            "Phase 3 — Joint fine-tuning",
            cfg.training.joint.epochs,
            cfg.training.joint.batch_size,
            1.10,
        ),
    ]

    for phase_name, epochs, batch_size, sec_per_batch in phases:
        batches_per_epoch = max(1, math.ceil(training_samples / batch_size))
        total_batches = batches_per_epoch * epochs
        total_seconds = total_batches * sec_per_batch
        gpu_hours = total_seconds / 3600
        cost = gpu_hours * gpu_rate

        costs.append(TrainingPhaseCost(
            phase_name=phase_name,
            epochs=epochs,
            batch_size=batch_size,
            estimated_batches_per_epoch=batches_per_epoch,
            seconds_per_batch=sec_per_batch,
            total_gpu_hours=round(gpu_hours, 2),
            gpu_type=gpu_type,
            cost_usd=round(cost, 2),
        ))

    return costs


def estimate_storage(cfg: FRLMConfig, total_chunks: int) -> List[StorageEstimate]:
    """Estimate disk-space requirements for each data component.

    All sizes in GB.
    """
    estimates: List[StorageEstimate] = []

    # Raw corpus (XML files: ~50 KB per document)
    corpus_docs = cfg.extraction.corpus.max_documents or 1000
    corpus_gb = corpus_docs * 50e-6  # 50 KB each
    estimates.append(StorageEstimate(
        component="Raw corpus (XML)",
        estimated_gb=round(corpus_gb, 3),
        monthly_cost_usd=round(corpus_gb * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    # Processed entity / relation JSON (~2 KB per chunk)
    proc_gb = total_chunks * 2e-6
    estimates.append(StorageEstimate(
        component="Processed entities/relations (JSON)",
        estimated_gb=round(proc_gb, 3),
        monthly_cost_usd=round(proc_gb * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    # Neo4j database (~10 bytes per triple, assume 5 triples/chunk)
    n_triples = total_chunks * 5
    neo4j_gb = n_triples * 10 / (1024**3)  # very rough
    neo4j_gb = max(neo4j_gb, 0.5)  # minimum overhead
    estimates.append(StorageEstimate(
        component="Neo4j knowledge graph",
        estimated_gb=round(neo4j_gb, 3),
        monthly_cost_usd=round(neo4j_gb * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    # FAISS index (768-dim float32 per vector)
    embedding_dim = cfg.sapbert.embedding_dim
    vectors = total_chunks * 5  # one per KG fact
    # IVF-PQ compresses to ~embedding_dim/4 bytes per vector
    faiss_bytes = vectors * (embedding_dim // 4)
    faiss_gb = faiss_bytes / (1024**3)
    faiss_gb = max(faiss_gb, 0.01)
    estimates.append(StorageEstimate(
        component="FAISS vector index",
        estimated_gb=round(faiss_gb, 3),
        monthly_cost_usd=round(faiss_gb * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    # Router labels (~500 bytes per chunk)
    labels_gb = total_chunks * 500 / (1024**3)
    estimates.append(StorageEstimate(
        component="Router labels (JSON)",
        estimated_gb=round(labels_gb, 3),
        monthly_cost_usd=round(labels_gb * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    # Model checkpoints (~1.5 GB per checkpoint, 3 phases × max_checkpoints)
    max_ckpt = cfg.training.max_checkpoints
    ckpt_gb = 3 * max_ckpt * 1.5
    estimates.append(StorageEstimate(
        component="Model checkpoints",
        estimated_gb=round(ckpt_gb, 2),
        monthly_cost_usd=round(ckpt_gb * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    # Logs and cache (misc, ~200 MB)
    estimates.append(StorageEstimate(
        component="Logs and cache",
        estimated_gb=0.2,
        monthly_cost_usd=round(0.2 * STORAGE_COST_PER_GB_MONTH, 2),
    ))

    return estimates


# ===========================================================================
# Report builder
# ===========================================================================


def build_cost_report(
    cfg: FRLMConfig,
    *,
    max_documents: Optional[int] = None,
    gpu_type: str = "A100-80GB",
) -> CostReport:
    """Build a full cost report from the configuration.

    Parameters
    ----------
    cfg : FRLMConfig
        Loaded FRLM configuration.
    max_documents : int, optional
        Override ``corpus.max_documents`` for what-if analysis.
    gpu_type : str
        GPU type for training cost estimation.

    Returns
    -------
    CostReport
    """
    total_chunks = estimate_corpus_chunks(cfg, max_documents=max_documents)

    api_costs = estimate_api_costs(cfg, total_chunks)
    training_costs = estimate_training_costs(cfg, total_chunks, gpu_type)
    storage_estimates = estimate_storage(cfg, total_chunks)

    total_api = sum(c.total_cost_usd for c in api_costs)
    total_training = sum(c.cost_usd for c in training_costs)
    total_storage = sum(s.monthly_cost_usd for s in storage_estimates)

    return CostReport(
        api_costs=api_costs,
        training_costs=training_costs,
        storage_estimates=storage_estimates,
        total_api_cost_usd=total_api,
        total_training_cost_usd=total_training,
        total_storage_monthly_usd=total_storage,
        grand_total_usd=total_api + total_training + total_storage,
    )


# ===========================================================================
# Pretty printer
# ===========================================================================


def print_cost_report(report: CostReport) -> None:
    """Print a human-readable cost report to stdout."""
    w = 68

    print()
    print("=" * w)
    print("  FRLM PIPELINE — COST ESTIMATE")
    print("=" * w)

    # --- Claude API ---
    print()
    print("  📡  CLAUDE API COSTS")
    print("  " + "-" * (w - 4))
    for c in report.api_costs:
        print(f"  {c.step_name}")
        print(f"    Model:          {c.model}")
        print(f"    Chunks:         {c.total_chunks:>12,}")
        print(f"    Input tokens:   {c.total_input_tokens:>12,}")
        print(f"    Output tokens:  {c.total_output_tokens:>12,}")
        print(f"    Input cost:     ${c.input_cost_usd:>10,.2f}")
        print(f"    Output cost:    ${c.output_cost_usd:>10,.2f}")
        print(f"    Subtotal:       ${c.total_cost_usd:>10,.2f}")
        print(f"    Rate limit:     {c.rate_limit_rpm} RPM")
        print(f"    Est. time:      {c.estimated_minutes:,.0f} min "
              f"(~{c.estimated_minutes / 60:.1f} hrs)")
        print()
    print(f"  {'API TOTAL:':<20} ${report.total_api_cost_usd:>10,.2f}")
    print()

    # --- Training ---
    print("  🖥️   GPU TRAINING COSTS")
    print("  " + "-" * (w - 4))
    for t in report.training_costs:
        print(f"  {t.phase_name}")
        print(f"    Epochs:         {t.epochs}")
        print(f"    Batch size:     {t.batch_size}")
        print(f"    Batches/epoch:  {t.estimated_batches_per_epoch:>12,}")
        print(f"    Sec/batch:      {t.seconds_per_batch}")
        print(f"    GPU hours:      {t.total_gpu_hours:>10,.2f}")
        print(f"    GPU type:       {t.gpu_type}")
        print(f"    Subtotal:       ${t.cost_usd:>10,.2f}")
        print()
    print(f"  {'TRAINING TOTAL:':<20} ${report.total_training_cost_usd:>10,.2f}")
    print()

    # --- Storage ---
    print("  💾  STORAGE ESTIMATES")
    print("  " + "-" * (w - 4))
    total_gb = 0.0
    for s in report.storage_estimates:
        total_gb += s.estimated_gb
        print(f"    {s.component:<40} {s.estimated_gb:>8.3f} GB "
              f"  ${s.monthly_cost_usd:.2f}/mo")
    print(f"    {'TOTAL':<40} {total_gb:>8.3f} GB "
          f"  ${report.total_storage_monthly_usd:.2f}/mo")
    print()

    # --- Grand total ---
    print("=" * w)
    print(f"  {'GRAND TOTAL (one-time):':<30} ${report.grand_total_usd:>10,.2f}")
    print(f"  {'  API costs:':<30} ${report.total_api_cost_usd:>10,.2f}")
    print(f"  {'  Training costs:':<30} ${report.total_training_cost_usd:>10,.2f}")
    print(f"  {'  Storage (first month):':<30} ${report.total_storage_monthly_usd:>10,.2f}")
    print("=" * w)
    print()


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Estimate resource costs for the FRLM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Override corpus.max_documents for what-if analysis",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="A100-80GB",
        choices=list(GPU_HOURLY_RATES.keys()),
        help="GPU type for training cost estimation (default: A100-80GB)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Export cost report as JSON to the given path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry-point for the cost estimation script.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (always 0).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)

    report = build_cost_report(
        cfg,
        max_documents=args.max_documents,
        gpu_type=args.gpu_type,
    )

    print_cost_report(report)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Cost report saved to %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
