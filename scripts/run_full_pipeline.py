#!/usr/bin/env python3
"""
run_full_pipeline.py — Master orchestration script for the FRLM pipeline.

Runs all 12 pipeline steps sequentially with per-step timing, skip-if-exists
logic, cost estimation prompts before Claude-API steps, ``--start-from``
resume support, and ``--dry-run`` mode.

Steps
-----
 1  Download corpus          (01_download_corpus.py)
 2  Extract entities         (02_extract_entities.py)
 3  Extract relations        (03_extract_relations.py)     [Claude API]
 4  Populate knowledge graph (04_populate_kg.py)
 5  Build FAISS index        (05_build_faiss_index.py)
 6  Generate router labels   (06_generate_router_labels.py) [Claude API]
 7  Prepare training data    (06b_prepare_training_data.py)
 8  Train router head        (07_train_router.py)
 9  Train retrieval head     (08_train_retrieval.py)
10  Joint fine-tuning        (09_train_joint.py)
11  Evaluate                 (10_evaluate.py)
12  Run inference            (11_run_inference.py)

Usage
-----
    # Run the full pipeline
    python scripts/run_full_pipeline.py --config config/default.yaml

    # Resume from step 7
    python scripts/run_full_pipeline.py --config config/default.yaml --start-from 7

    # Dry run — just print what would execute
    python scripts/run_full_pipeline.py --config config/default.yaml --dry-run

    # Non-interactive (skip cost confirmation prompts)
    python scripts/run_full_pipeline.py --config config/default.yaml --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import FRLMConfig, get_secret, load_config, setup_logging
from src.status import PipelineStatusTracker, scan_artifacts_into_status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_STEPS = 12
SCRIPTS_DIR = Path(__file__).resolve().parent


class StepStatus(str, Enum):
    """Status of a single pipeline step."""

    PENDING = "pending"
    SKIPPED = "skipped"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DRY_RUN = "dry_run"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    step_number: int
    name: str
    status: StepStatus
    elapsed_seconds: float = 0.0
    message: str = ""


@dataclass
class PipelineResult:
    """Aggregate result for the entire pipeline run."""

    steps: List[StepResult] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    start_from: int = 1

    @property
    def succeeded(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.steps),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 2),
            "start_from": self.start_from,
            "steps": [
                {
                    "step": s.step_number,
                    "name": s.name,
                    "status": s.status.value,
                    "elapsed_seconds": round(s.elapsed_seconds, 2),
                    "message": s.message,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Output-existence checks (for skip logic)
# ---------------------------------------------------------------------------


def _has_files(directory: Path, pattern: str) -> bool:
    """Return True if *directory* contains at least one file matching *pattern*."""
    if not directory.exists():
        return False
    return any(directory.glob(pattern))


def _dir_not_empty(directory: Path) -> bool:
    """Return True if *directory* exists and contains at least one file."""
    if not directory.exists():
        return False
    return any(directory.iterdir())


def _step7_complete(paths) -> bool:
    """Step 7 is complete only when tokenized files cover all label files."""
    tok_dir = paths.resolve("labels_dir") / "tokenized"
    lab_dir = paths.resolve("labels_dir")
    if not tok_dir.exists():
        return False
    n_tok = len(list(tok_dir.glob("router_PMC*.jsonl")))
    n_lab = len(list(lab_dir.glob("labels_PMC*.json")))
    # Complete if at least 90% of label files have been tokenized
    return n_lab > 0 and n_tok >= n_lab * 0.9


def check_step_complete(step: int, cfg: FRLMConfig) -> bool:
    """Heuristic check: does step *step* have output artefacts already?

    Returns True if the step appears to have been completed previously.
    This lets the pipeline skip already-done work on resume.
    """
    paths = cfg.paths
    # Resolve training output_dir relative to the project root so the
    # check works regardless of the current working directory.
    _output_base = paths.resolve("checkpoints_dir")
    checks: Dict[int, Callable[[], bool]] = {
        1: lambda: _has_files(paths.resolve("corpus_dir"), "*.xml"),
        2: lambda: _has_files(paths.resolve("processed_dir"), "entities_*.json"),
        3: lambda: _has_files(paths.resolve("processed_dir"), "relations_*.json"),
        4: lambda: (
            paths.resolve("kg_dir").exists()
            and _has_files(paths.resolve("kg_dir"), "exported_facts*")
        ),
        5: lambda: _dir_not_empty(paths.resolve("faiss_index_dir")),
        6: lambda: _dir_not_empty(paths.resolve("labels_dir")),
        7: lambda: _step7_complete(paths),
        8: lambda: _dir_not_empty(_output_base / "phase1_router"),
        9: lambda: _dir_not_empty(_output_base / "phase2_retrieval"),
        10: lambda: _dir_not_empty(_output_base / "phase3_joint"),
        11: lambda: _has_files(paths.resolve("export_dir"), "eval_results*"),
        12: lambda: _has_files(paths.resolve("export_dir"), "inference_results*"),
    }
    checker = checks.get(step)
    if checker is None:
        return False
    try:
        return checker()
    except Exception:
        return False


def get_pipeline_status(cfg: FRLMConfig) -> Dict[int, bool]:
    """Return completion status for every pipeline step."""
    return {step: check_step_complete(step, cfg) for step in range(1, TOTAL_STEPS + 1)}


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEP_NAMES: Dict[int, str] = {
    1: "Download corpus",
    2: "Extract entities",
    3: "Extract relations",
    4: "Populate knowledge graph",
    5: "Build FAISS index",
    6: "Generate router labels",
    7: "Prepare training data",
    8: "Train router head",
    9: "Train retrieval head",
    10: "Joint fine-tuning",
    11: "Evaluate",
    12: "Run inference",
}

STEP_SCRIPTS: Dict[int, str] = {
    1: "01_download_corpus.py",
    2: "02_extract_entities.py",
    3: "03_extract_relations.py",
    4: "04_populate_kg.py",
    5: "05_build_faiss_index.py",
    6: "06_generate_router_labels.py",
    7: "06b_prepare_training_data.py",
    8: "07_train_router.py",
    9: "08_train_retrieval.py",
    10: "09_train_joint.py",
    11: "10_evaluate.py",
    12: "11_run_inference.py",
}

# Steps that use the Claude API and thus incur cost
CLAUDE_API_STEPS = {3, 6}


# ---------------------------------------------------------------------------
# Cost estimation prompt
# ---------------------------------------------------------------------------


def _estimate_claude_cost(step: int, cfg: FRLMConfig) -> Dict[str, Any]:
    """Quick cost estimate for Claude-API pipeline steps.

    Step 3 (relation extraction):
      - Each entity-file is sent to Claude for relation mining.
      - Heuristic: ~2 000 input tokens per entity chunk + 500 output tokens.

    Step 6 (router labeling):
      - Each text chunk is labelled as factual / linguistic.
      - Heuristic: ~1 800 input tokens per chunk + 450 output tokens.
    """
    if step not in CLAUDE_API_STEPS:
        return {"error": f"Step {step} is not a Claude API step"}

    corpus_cfg = cfg.extraction.corpus
    max_docs = corpus_cfg.max_documents if corpus_cfg.max_documents is not None else 1000
    chunk_size = corpus_cfg.chunk_size

    # Rough: each doc produces ~(doc_tokens / chunk_size) chunks
    avg_doc_tokens = 4000  # typical PMC OA abstract+body
    chunks_per_doc = max(1, avg_doc_tokens // chunk_size)
    total_chunks = max_docs * chunks_per_doc

    if step == 3:
        model = cfg.extraction.relation.model
        input_per_call = 2000
        output_per_call = 500
    else:  # step == 6
        model = cfg.labeling.model
        input_per_call = 1800
        output_per_call = 450

    total_input = total_chunks * input_per_call
    total_output = total_chunks * output_per_call

    # Sonnet pricing: $3/M input, $15/M output
    input_cost = total_input * 3.00 / 1_000_000
    output_cost = total_output * 15.00 / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "step": step,
        "name": STEP_NAMES[step],
        "model": model,
        "estimated_chunks": total_chunks,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 2),
    }


def _prompt_cost_confirmation(
    estimate: Dict[str, Any], *, auto_yes: bool = False
) -> bool:
    """Show a cost estimate and ask for user confirmation.

    Returns True if the user confirms (or ``auto_yes`` is set).
    """
    print("\n" + "=" * 60)
    print(f"  COST ESTIMATE — Step {estimate['step']}: {estimate['name']}")
    print("=" * 60)
    print(f"  Model:           {estimate['model']}")
    print(f"  Chunks:          {estimate['estimated_chunks']:,}")
    print(f"  Input tokens:    {estimate['estimated_input_tokens']:,}")
    print(f"  Output tokens:   {estimate['estimated_output_tokens']:,}")
    print(f"  Estimated cost:  ${estimate['estimated_cost_usd']:.2f}")
    print("=" * 60)

    if auto_yes:
        print("  (--yes flag set — proceeding automatically)")
        return True

    try:
        answer = input("  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Pre-flight checks — validate services & env vars before any step runs
# ---------------------------------------------------------------------------


def preflight_checks(
    cfg: FRLMConfig,
    start_from: int = 1,
    stop_after: int = TOTAL_STEPS,
) -> List[str]:
    """Verify that required services and credentials are available.

    Checks are scoped to the steps that will actually run (``start_from``
    through ``stop_after``).

    Returns a list of human-readable error strings.  An empty list means
    all checks passed.
    """
    errors: List[str] = []
    planned_steps = set(range(start_from, stop_after + 1))

    # -- Steps that need Neo4j: 4, 5, 11, 12 --------------------------------
    NEO4J_STEPS = {4, 5, 11, 12}
    if planned_steps & NEO4J_STEPS:
        # 1. Check password is not the placeholder
        neo4j_cfg = cfg.neo4j
        if neo4j_cfg.password in ("CHANGE_ME", ""):
            errors.append(
                "Neo4j password is not set. "
                "Set neo4j.password in config/secrets.properties or update config/default.yaml"
            )

        # 2. Check connectivity + auth
        try:
            from neo4j import GraphDatabase
            from neo4j.exceptions import AuthError, ServiceUnavailable

            driver = GraphDatabase.driver(
                neo4j_cfg.uri,
                auth=(neo4j_cfg.username, neo4j_cfg.password),
            )
            try:
                driver.verify_connectivity()
            except AuthError:
                errors.append(
                    f"Neo4j authentication failed (uri={neo4j_cfg.uri}, "
                    f"user={neo4j_cfg.username}). Check config/secrets.properties."
                )
            except ServiceUnavailable:
                errors.append(
                    f"Cannot reach Neo4j at {neo4j_cfg.uri}. "
                    "Is the server running?  (try: make neo4j-status)"
                )
            except Exception as exc:
                errors.append(f"Neo4j connection error: {exc}")
            finally:
                driver.close()
        except ImportError:
            errors.append(
                "Python 'neo4j' package is not installed. Run: pip install neo4j"
            )

    # -- Steps that need Anthropic API: 3, 6 ---------------------------------
    CLAUDE_STEPS = {3, 6}
    if planned_steps & CLAUDE_STEPS:
        api_key = get_secret("anthropic.api_key", cfg.extraction.relation.api_key)
        if not api_key or api_key in ("CHANGE_ME", ""):
            errors.append(
                "Anthropic API key is not set. "
                "Set anthropic.api_key in config/secrets.properties"
            )

    # -- Steps that need W&B: 7, 8, 9 ----------------------------------------
    WANDB_STEPS = {7, 8, 9}
    if planned_steps & WANDB_STEPS and cfg.wandb.enabled:
        api_key = get_secret("wandb.api_key", cfg.wandb.api_key)
        if not api_key or api_key in ("CHANGE_ME", ""):
            errors.append(
                "W&B API key is not set but wandb.enabled=true. "
                "Set wandb.api_key in config/secrets.properties or set wandb.enabled=false"
            )

    return errors


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def run_step(
    step: int,
    cfg: FRLMConfig,
    *,
    config_path: str,
    dry_run: bool = False,
    skip_completed: bool = True,
    auto_yes: bool = False,
    extra_args: Optional[Sequence[str]] = None,
    tracker: Optional[PipelineStatusTracker] = None,
) -> StepResult:
    """Execute a single pipeline step.

    Parameters
    ----------
    step : int
        Step number (1–12).
    cfg : FRLMConfig
        Loaded configuration.
    config_path : str
        Path to the YAML config file (forwarded to the sub-script).
    dry_run : bool
        If True, just log what would be done.
    skip_completed : bool
        If True, skip steps whose output artefacts already exist.
    auto_yes : bool
        If True, skip interactive cost-confirmation prompts.
    extra_args : sequence of str, optional
        Additional CLI arguments forwarded to the sub-script.
    tracker : PipelineStatusTracker, optional
        Centralized status tracker (updated on every state transition).

    Returns
    -------
    StepResult
    """
    name = STEP_NAMES.get(step, f"Unknown step {step}")
    script = STEP_SCRIPTS.get(step)

    if script is None:
        if tracker:
            tracker.mark_failed(step, f"No script registered for step {step}")
        return StepResult(
            step_number=step,
            name=name,
            status=StepStatus.FAILED,
            message=f"No script registered for step {step}",
        )

    # --- skip-if-exists ---
    if skip_completed and check_step_complete(step, cfg):
        logger.info(
            "[Step %2d/%d] %-28s — SKIP (output exists)", step, TOTAL_STEPS, name
        )
        if tracker:
            tracker.mark_skipped(step, "output artefacts already exist")
        return StepResult(
            step_number=step,
            name=name,
            status=StepStatus.SKIPPED,
            message="Output artefacts already exist",
        )

    # --- dry-run ---
    if dry_run:
        cmd_preview = f"{sys.executable} {SCRIPTS_DIR / script} --config {config_path}"
        logger.info(
            "[Step %2d/%d] %-28s — DRY RUN: %s",
            step,
            TOTAL_STEPS,
            name,
            cmd_preview,
        )
        return StepResult(
            step_number=step,
            name=name,
            status=StepStatus.DRY_RUN,
            message=cmd_preview,
        )

    # --- cost prompt for Claude API steps ---
    if step in CLAUDE_API_STEPS:
        estimate = _estimate_claude_cost(step, cfg)
        if not _prompt_cost_confirmation(estimate, auto_yes=auto_yes):
            logger.warning(
                "[Step %2d/%d] %-28s — SKIPPED by user (cost declined)",
                step,
                TOTAL_STEPS,
                name,
            )
            if tracker:
                tracker.mark_skipped(step, "user declined cost estimate")
            return StepResult(
                step_number=step,
                name=name,
                status=StepStatus.SKIPPED,
                message="User declined cost estimate",
            )

    # --- execute ---
    script_path = SCRIPTS_DIR / script
    cmd: List[str] = [sys.executable, str(script_path), "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(
        "[Step %2d/%d] %-28s — RUNNING  %s",
        step,
        TOTAL_STEPS,
        name,
        " ".join(cmd),
    )

    if tracker:
        tracker.mark_running(step)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            text=True,
            capture_output=False,
        )
        elapsed = time.time() - t0
        logger.info(
            "[Step %2d/%d] %-28s — DONE in %.1fs",
            step,
            TOTAL_STEPS,
            name,
            elapsed,
        )
        if tracker:
            tracker.mark_completed(step)
            # Back-fill item counts from artifacts after completion
            scan_artifacts_into_status(tracker, cfg)
        return StepResult(
            step_number=step,
            name=name,
            status=StepStatus.COMPLETED,
            elapsed_seconds=elapsed,
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        logger.error(
            "[Step %2d/%d] %-28s — FAILED (exit code %d) after %.1fs",
            step,
            TOTAL_STEPS,
            name,
            exc.returncode,
            elapsed,
        )
        if tracker:
            tracker.mark_failed(step, f"Exit code {exc.returncode}")
            # Still scan for partial progress
            scan_artifacts_into_status(tracker, cfg)
        return StepResult(
            step_number=step,
            name=name,
            status=StepStatus.FAILED,
            elapsed_seconds=elapsed,
            message=f"Exit code {exc.returncode}",
        )
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error(
            "[Step %2d/%d] %-28s — ERROR: %s", step, TOTAL_STEPS, name, exc
        )
        if tracker:
            tracker.mark_failed(step, str(exc))
        return StepResult(
            step_number=step,
            name=name,
            status=StepStatus.FAILED,
            elapsed_seconds=elapsed,
            message=str(exc),
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: FRLMConfig,
    *,
    config_path: str,
    start_from: int = 1,
    stop_after: Optional[int] = None,
    dry_run: bool = False,
    skip_completed: bool = True,
    auto_yes: bool = False,
    stop_on_failure: bool = True,
) -> PipelineResult:
    """Run the FRLM pipeline from *start_from* to *stop_after* (inclusive).

    Parameters
    ----------
    cfg : FRLMConfig
        Loaded FRLM configuration.
    config_path : str
        Path to YAML config (forwarded to sub-scripts).
    start_from : int
        First step to execute (1–12).
    stop_after : int, optional
        Last step to execute (default: 12).
    dry_run : bool
        Print commands without executing.
    skip_completed : bool
        Skip steps whose outputs already exist.
    auto_yes : bool
        Skip cost-confirmation prompts.
    stop_on_failure : bool
        Abort the pipeline on the first failure.

    Returns
    -------
    PipelineResult
    """
    if stop_after is None:
        stop_after = TOTAL_STEPS

    start_from = max(1, min(start_from, TOTAL_STEPS))
    stop_after = max(start_from, min(stop_after, TOTAL_STEPS))

    # --- pre-flight checks ---
    if not dry_run:
        logger.info("Running pre-flight checks for steps %d → %d ...", start_from, stop_after)
        pf_errors = preflight_checks(cfg, start_from, stop_after)
        if pf_errors:
            logger.error("=" * 70)
            logger.error("  PRE-FLIGHT CHECKS FAILED")
            logger.error("=" * 70)
            for i, err in enumerate(pf_errors, 1):
                logger.error("  %d. %s", i, err)
            logger.error("=" * 70)
            logger.error("Fix the above issues and re-run the pipeline.")
            pipeline = PipelineResult(start_from=start_from)
            pipeline.steps.append(
                StepResult(
                    step_number=0,
                    name="Pre-flight checks",
                    status=StepStatus.FAILED,
                    message="; ".join(pf_errors),
                )
            )
            return pipeline
        logger.info("Pre-flight checks passed ✓")

    pipeline = PipelineResult(start_from=start_from)
    tracker = PipelineStatusTracker()
    # Back-fill status from any existing artifacts
    scan_artifacts_into_status(tracker, cfg)

    logger.info("=" * 70)
    logger.info("  FRLM Pipeline — Steps %d → %d  (dry_run=%s)", start_from, stop_after, dry_run)
    logger.info("=" * 70)

    t_pipeline_start = time.time()

    for step in range(start_from, stop_after + 1):
        result = run_step(
            step,
            cfg,
            config_path=config_path,
            dry_run=dry_run,
            skip_completed=skip_completed,
            auto_yes=auto_yes,
            tracker=tracker,
        )
        pipeline.steps.append(result)

        if result.status == StepStatus.FAILED and stop_on_failure:
            logger.error(
                "Pipeline aborted at step %d (%s). Use --start-from %d to resume.",
                step,
                result.name,
                step,
            )
            break

    pipeline.total_elapsed_seconds = time.time() - t_pipeline_start

    # --- Summary ---
    logger.info("=" * 70)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "  Completed: %d | Skipped: %d | Failed: %d | Total time: %.1fs",
        pipeline.succeeded,
        pipeline.skipped,
        pipeline.failed,
        pipeline.total_elapsed_seconds,
    )
    for s in pipeline.steps:
        status_icon = {
            StepStatus.COMPLETED: "✓",
            StepStatus.SKIPPED: "⏭",
            StepStatus.FAILED: "✗",
            StepStatus.DRY_RUN: "⊘",
            StepStatus.PENDING: "·",
            StepStatus.RUNNING: "▶",
        }.get(s.status, "?")
        logger.info(
            "  %s  Step %2d  %-28s  %8.1fs  %s",
            status_icon,
            s.step_number,
            s.name,
            s.elapsed_seconds,
            s.message,
        )
    logger.info("=" * 70)

    return pipeline


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


def print_status(cfg: FRLMConfig) -> None:
    """Print current pipeline completion status to stdout.

    Uses the centralized :class:`PipelineStatusTracker` and scans the
    filesystem for artifacts so the display always reflects reality.
    """
    tracker = PipelineStatusTracker()
    scan_artifacts_into_status(tracker, cfg)
    tracker.print_status()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the master pipeline script."""
    parser = argparse.ArgumentParser(
        description="FRLM master pipeline orchestrator — run all 12 steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_full_pipeline.py --config config/default.yaml
  python scripts/run_full_pipeline.py --config config/default.yaml --start-from 7
  python scripts/run_full_pipeline.py --config config/default.yaml --dry-run
  python scripts/run_full_pipeline.py --status
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        choices=range(1, TOTAL_STEPS + 1),
        metavar="N",
        help="Step to start/resume from (1–12, default: 1)",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        choices=range(1, TOTAL_STEPS + 1),
        metavar="N",
        help="Step to stop after (1–12, default: 12)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run steps even if output artefacts exist",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip interactive cost-confirmation prompts",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue running subsequent steps even if one fails",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print pipeline completion status and exit",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Save a JSON run report to the given path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry-point for the master pipeline script.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(cfg)

    # --- status mode ---
    if args.status:
        print_status(cfg)
        return 0

    # --- run pipeline ---
    result = run_pipeline(
        cfg,
        config_path=args.config,
        start_from=args.start_from,
        stop_after=args.stop_after,
        dry_run=args.dry_run,
        skip_completed=not args.no_skip,
        auto_yes=args.yes,
        stop_on_failure=not args.no_stop_on_failure,
    )

    # --- optional report ---
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result.summary_dict(), f, indent=2)
        logger.info("Run report saved to %s", report_path)

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
