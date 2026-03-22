"""
Centralized pipeline status tracker for FRLM.

Persists a ``pipeline_status.json`` file that records:
* Per-step completion state (pending / running / completed / failed / partial)
* Start / end timestamps, durations
* Item-level progress within steps (e.g. 14/99 files labelled)
* Training-specific metadata (epoch, global_step, best_metric)
* Error history (last error message per step)
* Accumulated API costs

Usage::

    from src.status import PipelineStatusTracker

    tracker = PipelineStatusTracker()           # loads existing status
    tracker.mark_running(step=6, total_items=99)
    tracker.update_progress(step=6, completed_items=14)
    tracker.mark_completed(step=6)
    tracker.save()

    # Quick check from CLI:
    #   python -m src.status                     # prints table
    #   python -m src.status --json              # dumps JSON
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATUS_PATH = _PROJECT_ROOT / "pipeline_status.json"

TOTAL_STEPS = 12

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


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blank_step(step: int) -> Dict[str, Any]:
    """Return a fresh status record for a single step."""
    return {
        "step": step,
        "name": STEP_NAMES.get(step, f"Step {step}"),
        "status": "pending",              # pending | running | completed | failed | partial
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "total_items": None,              # e.g. total files to process
        "completed_items": 0,             # files / docs processed so far
        "skipped_items": 0,               # items skipped (already existed)
        "failed_items": 0,                # items that errored
        "last_error": None,
        "cost_usd": 0.0,                  # accumulated API cost (Claude)
        "training": None,                 # dict with epoch, global_step, best_metric, checkpoint_dir
        "extra": {},                      # any other metadata
    }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PipelineStatusTracker:
    """Read / write ``pipeline_status.json`` with fine-grained progress."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_STATUS_PATH
        self._data: Dict[str, Any] = self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                logger.debug("Loaded pipeline status from %s", self._path)
                # Ensure all 12 steps exist (forward-compat)
                steps = {s["step"]: s for s in data.get("steps", [])}
                for i in range(1, TOTAL_STEPS + 1):
                    if i not in steps:
                        steps[i] = _blank_step(i)
                data["steps"] = [steps[i] for i in range(1, TOTAL_STEPS + 1)]
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Corrupt status file, starting fresh: %s", exc)
        return self._blank()

    def _blank(self) -> Dict[str, Any]:
        return {
            "project": "frlm",
            "last_updated": _now_iso(),
            "steps": [_blank_step(i) for i in range(1, TOTAL_STEPS + 1)],
        }

    def save(self) -> None:
        """Atomically persist status to disk."""
        self._data["last_updated"] = _now_iso()
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)
        tmp.replace(self._path)
        logger.debug("Pipeline status saved to %s", self._path)

    # -- step accessors -----------------------------------------------------

    def _step(self, step: int) -> Dict[str, Any]:
        return self._data["steps"][step - 1]

    # -- state transitions --------------------------------------------------

    def mark_running(
        self,
        step: int,
        *,
        total_items: Optional[int] = None,
    ) -> None:
        """Mark *step* as running."""
        s = self._step(step)
        s["status"] = "running"
        s["started_at"] = _now_iso()
        s["finished_at"] = None
        if total_items is not None:
            s["total_items"] = total_items
        self.save()

    def mark_completed(self, step: int) -> None:
        """Mark *step* as completed."""
        s = self._step(step)
        s["status"] = "completed"
        s["finished_at"] = _now_iso()
        if s["started_at"]:
            try:
                t0 = datetime.fromisoformat(s["started_at"])
                t1 = datetime.fromisoformat(s["finished_at"])
                s["elapsed_seconds"] = round((t1 - t0).total_seconds(), 2)
            except Exception:
                pass
        self.save()

    def mark_failed(self, step: int, error: str = "") -> None:
        """Mark *step* as failed."""
        s = self._step(step)
        s["status"] = "failed"
        s["finished_at"] = _now_iso()
        s["last_error"] = error
        if s["started_at"]:
            try:
                t0 = datetime.fromisoformat(s["started_at"])
                t1 = datetime.fromisoformat(s["finished_at"])
                s["elapsed_seconds"] = round((t1 - t0).total_seconds(), 2)
            except Exception:
                pass
        self.save()

    def mark_partial(self, step: int) -> None:
        """Mark *step* as partially completed (interrupted)."""
        s = self._step(step)
        s["status"] = "partial"
        s["finished_at"] = _now_iso()
        if s["started_at"]:
            try:
                t0 = datetime.fromisoformat(s["started_at"])
                t1 = datetime.fromisoformat(s["finished_at"])
                s["elapsed_seconds"] = round((t1 - t0).total_seconds(), 2)
            except Exception:
                pass
        self.save()

    def mark_skipped(self, step: int, message: str = "output exists") -> None:
        """Mark *step* as skipped."""
        s = self._step(step)
        s["status"] = "skipped"
        s["finished_at"] = _now_iso()
        s["extra"]["skip_reason"] = message
        self.save()

    # -- progress updates ---------------------------------------------------

    def update_progress(
        self,
        step: int,
        *,
        completed_items: Optional[int] = None,
        skipped_items: Optional[int] = None,
        failed_items: Optional[int] = None,
        total_items: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """Update item-level progress counters (does NOT auto-save for perf)."""
        s = self._step(step)
        if completed_items is not None:
            s["completed_items"] = completed_items
        if skipped_items is not None:
            s["skipped_items"] = skipped_items
        if failed_items is not None:
            s["failed_items"] = failed_items
        if total_items is not None:
            s["total_items"] = total_items
        if cost_usd is not None:
            s["cost_usd"] = round(cost_usd, 6)

    def update_training(
        self,
        step: int,
        *,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        best_metric: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update training-specific metadata for a training step."""
        s = self._step(step)
        if s["training"] is None:
            s["training"] = {}
        t = s["training"]
        if epoch is not None:
            t["epoch"] = epoch
        if global_step is not None:
            t["global_step"] = global_step
        if best_metric is not None:
            t["best_metric"] = round(best_metric, 6)
        if checkpoint_dir is not None:
            t["checkpoint_dir"] = checkpoint_dir
        if metrics is not None:
            t["metrics"] = {k: round(v, 6) for k, v in metrics.items()}

    def set_extra(self, step: int, key: str, value: Any) -> None:
        """Set an arbitrary metadata field on a step."""
        self._step(step)["extra"][key] = value

    # -- queries ------------------------------------------------------------

    def get_step(self, step: int) -> Dict[str, Any]:
        """Return a copy of the status record for *step*."""
        return dict(self._step(step))

    def get_all(self) -> Dict[str, Any]:
        """Return the full status dict (deep copy)."""
        return json.loads(json.dumps(self._data))

    def is_complete(self, step: int) -> bool:
        return self._step(step)["status"] == "completed"

    def progress_fraction(self, step: int) -> Optional[float]:
        """Return completed_items / total_items, or None if unknown."""
        s = self._step(step)
        total = s.get("total_items")
        if total and total > 0:
            return s["completed_items"] / total
        return None

    # -- display ------------------------------------------------------------

    def print_status(self) -> None:
        """Pretty-print the pipeline status table."""
        STATUS_ICONS = {
            "pending": "·",
            "running": "▶",
            "completed": "✓",
            "failed": "✗",
            "partial": "◐",
            "skipped": "⏭",
        }
        print()
        print("  FRLM Pipeline Status")
        print("  " + "─" * 68)
        print(f"  {'':2}  {'Step':5} {'Name':28} {'Status':9} {'Progress':14} {'Time':>8}")
        print("  " + "─" * 68)

        for s in self._data["steps"]:
            icon = STATUS_ICONS.get(s["status"], "?")
            status_str = s["status"]

            # Progress string
            total = s.get("total_items")
            done = s.get("completed_items", 0)
            if total and total > 0:
                pct = done / total * 100
                progress = f"{done}/{total} ({pct:.0f}%)"
            elif s["status"] == "completed":
                progress = "done"
            else:
                progress = ""

            # Time string
            elapsed = s.get("elapsed_seconds", 0)
            if elapsed > 0:
                if elapsed > 3600:
                    time_str = f"{elapsed / 3600:.1f}h"
                elif elapsed > 60:
                    time_str = f"{elapsed / 60:.1f}m"
                else:
                    time_str = f"{elapsed:.0f}s"
            else:
                time_str = ""

            print(
                f"  {icon}  {s['step']:2d}    {s['name']:28s} {status_str:9s} {progress:14s} {time_str:>8s}"
            )

            # Training sub-info
            if s.get("training"):
                t = s["training"]
                parts = []
                if "epoch" in t:
                    parts.append(f"epoch={t['epoch']}")
                if "global_step" in t:
                    parts.append(f"step={t['global_step']}")
                if "best_metric" in t:
                    parts.append(f"best={t['best_metric']:.4f}")
                if parts:
                    print(f"         {'':28s}           {' | '.join(parts)}")

            # Cost sub-info
            if s.get("cost_usd", 0) > 0:
                print(f"         {'':28s}           cost=${s['cost_usd']:.4f}")

            # Error sub-info
            if s.get("last_error"):
                err_preview = s["last_error"][:60]
                print(f"         {'':28s}           ⚠ {err_preview}")

        completed = sum(1 for s in self._data["steps"] if s["status"] == "completed")
        partial = sum(1 for s in self._data["steps"] if s["status"] == "partial")
        failed = sum(1 for s in self._data["steps"] if s["status"] == "failed")
        print("  " + "─" * 68)
        parts = [f"{completed}/{TOTAL_STEPS} completed"]
        if partial:
            parts.append(f"{partial} partial")
        if failed:
            parts.append(f"{failed} failed")
        print(f"  {' | '.join(parts)}")
        print(f"  Last updated: {self._data.get('last_updated', 'never')}")
        print()


# ---------------------------------------------------------------------------
# Convenience: scan artifacts and auto-populate status for past runs
# ---------------------------------------------------------------------------

def scan_artifacts_into_status(
    tracker: PipelineStatusTracker,
    cfg: Any,
) -> None:
    """Walk the filesystem and back-fill status for steps that have artifacts
    but no status entry (e.g. runs done before the tracker existed).

    This is a best-effort heuristic — it cannot recover timestamps, but it
    ensures the status file reflects reality.
    """
    from pathlib import Path

    paths = cfg.paths

    def _count_files(directory: Path, pattern: str) -> int:
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))

    def _dir_not_empty(directory: Path) -> bool:
        return directory.exists() and any(directory.iterdir())

    # Step 1: corpus XMLs
    corpus_dir = paths.resolve("corpus_dir")
    n_xml = _count_files(corpus_dir, "*.xml")
    if n_xml > 0 and not tracker.is_complete(1):
        tracker.update_progress(1, completed_items=n_xml, total_items=n_xml)
        tracker.mark_completed(1)

    # Step 2: entity JSONs
    processed_dir = paths.resolve("processed_dir")
    n_ent = _count_files(processed_dir, "entities_*.json")
    if n_ent > 0:
        s = tracker.get_step(2)
        tracker.update_progress(2, completed_items=n_ent, total_items=n_xml or n_ent)
        if n_ent >= (n_xml or n_ent) and not tracker.is_complete(2):
            tracker.mark_completed(2)
        elif s["status"] == "pending":
            tracker._step(2)["status"] = "partial"

    # Step 3: relation JSONs
    n_rel = _count_files(processed_dir, "relations_*.json")
    if n_rel > 0:
        s = tracker.get_step(3)
        tracker.update_progress(3, completed_items=n_rel, total_items=n_xml or n_rel)
        if n_rel >= (n_xml or n_rel) and not tracker.is_complete(3):
            tracker.mark_completed(3)
        elif s["status"] == "pending":
            tracker._step(3)["status"] = "partial"

    # Step 4: KG
    kg_dir = paths.resolve("kg_dir")
    if _count_files(kg_dir, "exported_facts*") > 0 and not tracker.is_complete(4):
        tracker.mark_completed(4)

    # Step 5: FAISS
    faiss_dir = paths.resolve("faiss_index_dir")
    if _dir_not_empty(faiss_dir) and not tracker.is_complete(5):
        n_idx = _count_files(faiss_dir, "*.faiss")
        tracker.update_progress(5, completed_items=n_idx)
        tracker.mark_completed(5)

    # Step 6: labels
    labels_dir = paths.resolve("labels_dir")
    n_labels = _count_files(labels_dir, "labels_*.json")
    if n_labels > 0:
        s = tracker.get_step(6)
        # Total expected = number of entity files (one label per entity file)
        expected = n_ent if n_ent > 0 else n_labels
        tracker.update_progress(6, completed_items=n_labels, total_items=expected)
        if n_labels >= expected and not tracker.is_complete(6):
            tracker.mark_completed(6)
        elif s["status"] == "pending":
            tracker._step(6)["status"] = "partial"

    # Step 7: tokenized training data
    tok_dir = labels_dir / "tokenized"
    n_tok = _count_files(tok_dir, "*.jsonl")
    if n_tok > 0:
        s = tracker.get_step(7)
        tracker.update_progress(7, completed_items=n_tok, total_items=n_labels or n_tok)
        if n_tok >= (n_labels or n_tok) and not tracker.is_complete(7):
            tracker.mark_completed(7)
        elif s["status"] == "pending":
            tracker._step(7)["status"] = "partial"

    # Steps 8-10: training checkpoints
    try:
        ckpt_base = Path(cfg.training.output_dir)
    except Exception:
        ckpt_base = paths.resolve("checkpoints_dir")

    for step_num, phase_dir in [(8, "phase1_router"), (9, "phase2_retrieval"), (10, "phase3_joint")]:
        phase_path = ckpt_base / phase_dir
        if phase_path.exists():
            ckpt_dirs = sorted([d for d in phase_path.iterdir() if d.is_dir()])
            if ckpt_dirs:
                latest = ckpt_dirs[-1]
                meta_path = latest / "meta.json"
                training_info = {}
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        state = meta.get("state", {})
                        training_info = {
                            "epoch": state.get("epoch"),
                            "global_step": state.get("global_step"),
                            "best_metric": state.get("best_metric"),
                            "checkpoint_dir": str(latest),
                        }
                        if meta.get("metrics"):
                            training_info["metrics"] = meta["metrics"]
                    except Exception:
                        pass

                s = tracker.get_step(step_num)
                tracker.update_progress(step_num, completed_items=len(ckpt_dirs))
                if training_info:
                    tracker.update_training(step_num, **training_info)
                if not tracker.is_complete(step_num):
                    # Heuristic: if there's a checkpoint, consider it complete
                    # (the individual script would have exited cleanly)
                    tracker._step(step_num)["status"] = "completed"

    # Step 11: eval results
    export_dir = paths.resolve("export_dir")
    if _count_files(export_dir, "eval_results*") > 0 and not tracker.is_complete(11):
        tracker.mark_completed(11)

    # Step 12: inference results
    if _count_files(export_dir, "inference_results*") > 0 and not tracker.is_complete(12):
        tracker.mark_completed(12)

    tracker.save()


# ---------------------------------------------------------------------------
# CLI entry-point:  python -m src.status
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(_PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="FRLM pipeline status viewer")
    parser.add_argument("--json", action="store_true", help="Dump raw JSON")
    parser.add_argument(
        "--scan", action="store_true",
        help="Scan filesystem artifacts and back-fill status",
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--path", default=None,
        help="Custom status file path (default: pipeline_status.json in project root)",
    )
    args = parser.parse_args()

    tracker = PipelineStatusTracker(path=args.path)

    if args.scan:
        from config.config import load_config
        cfg = load_config(args.config)
        scan_artifacts_into_status(tracker, cfg)

    if args.json:
        print(json.dumps(tracker.get_all(), indent=2))
    else:
        tracker.print_status()


if __name__ == "__main__":
    main()



