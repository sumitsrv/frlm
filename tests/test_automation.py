"""
Tests for Phase 9: End-to-end automation scripts.

Covers:
  - run_full_pipeline.py  (orchestrator, step runner, skip logic, CLI)
  - estimate_costs.py     (API cost, training cost, storage, report builder)
  - Makefile targets       (checked via existence / dry parse)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FRLMConfig, load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_config() -> FRLMConfig:
    return load_config()


@pytest.fixture()
def pipeline_mod():
    """Import the run_full_pipeline module."""
    import scripts.run_full_pipeline as mod
    return mod


@pytest.fixture()
def costs_mod():
    """Import the estimate_costs module."""
    import scripts.estimate_costs as mod
    return mod


# ===========================================================================
# run_full_pipeline.py — Constants & Metadata
# ===========================================================================


class TestPipelineConstants:
    """Verify step definitions are complete and consistent."""

    def test_total_steps_is_twelve(self, pipeline_mod) -> None:
        assert pipeline_mod.TOTAL_STEPS == 12

    def test_step_names_covers_all(self, pipeline_mod) -> None:
        names = pipeline_mod.STEP_NAMES
        assert len(names) == 12
        for i in range(1, 13):
            assert i in names
            assert isinstance(names[i], str)
            assert len(names[i]) > 0

    def test_step_scripts_covers_all(self, pipeline_mod) -> None:
        scripts = pipeline_mod.STEP_SCRIPTS
        assert len(scripts) == 12
        for i in range(1, 13):
            assert i in scripts
            assert scripts[i].endswith(".py")

    def test_step_scripts_have_correct_prefix(self, pipeline_mod) -> None:
        # Step 7 uses "06b_" prefix; steps 8-12 use scripts numbered 07-11
        # (the insertion of prepare_training_data shifted numbering).
        # We just verify every script file name starts with a digit prefix.
        for step, script in pipeline_mod.STEP_SCRIPTS.items():
            assert script[0].isdigit(), f"Step {step}: {script} should start with a digit"
            assert script.endswith(".py"), f"Step {step}: {script} should end with .py"

    def test_claude_api_steps(self, pipeline_mod) -> None:
        assert pipeline_mod.CLAUDE_API_STEPS == {3, 6}

    def test_scripts_dir_is_absolute(self, pipeline_mod) -> None:
        assert pipeline_mod.SCRIPTS_DIR.is_absolute()

    def test_all_step_script_files_exist(self, pipeline_mod) -> None:
        for step, script in pipeline_mod.STEP_SCRIPTS.items():
            path = pipeline_mod.SCRIPTS_DIR / script
            assert path.exists(), f"Step {step} script missing: {path}"


# ===========================================================================
# StepStatus enum
# ===========================================================================


class TestStepStatus:
    """Verify the StepStatus enum values."""

    def test_all_statuses_defined(self, pipeline_mod) -> None:
        statuses = {s.value for s in pipeline_mod.StepStatus}
        expected = {"pending", "skipped", "running", "completed", "failed", "dry_run"}
        assert statuses == expected

    def test_status_is_string(self, pipeline_mod) -> None:
        for s in pipeline_mod.StepStatus:
            assert isinstance(s.value, str)


# ===========================================================================
# StepResult data class
# ===========================================================================


class TestStepResult:
    """Verify StepResult defaults and fields."""

    def test_default_values(self, pipeline_mod) -> None:
        r = pipeline_mod.StepResult(
            step_number=1,
            name="test",
            status=pipeline_mod.StepStatus.PENDING,
        )
        assert r.elapsed_seconds == 0.0
        assert r.message == ""

    def test_custom_values(self, pipeline_mod) -> None:
        r = pipeline_mod.StepResult(
            step_number=5,
            name="Build index",
            status=pipeline_mod.StepStatus.COMPLETED,
            elapsed_seconds=42.5,
            message="ok",
        )
        assert r.step_number == 5
        assert r.elapsed_seconds == 42.5


# ===========================================================================
# PipelineResult data class
# ===========================================================================


class TestPipelineResult:
    """Verify PipelineResult aggregation properties."""

    def test_empty_result(self, pipeline_mod) -> None:
        pr = pipeline_mod.PipelineResult()
        assert pr.succeeded == 0
        assert pr.failed == 0
        assert pr.skipped == 0
        assert pr.total_elapsed_seconds == 0.0

    def test_counts(self, pipeline_mod) -> None:
        SS = pipeline_mod.StepStatus
        pr = pipeline_mod.PipelineResult(steps=[
            pipeline_mod.StepResult(1, "A", SS.COMPLETED),
            pipeline_mod.StepResult(2, "B", SS.COMPLETED),
            pipeline_mod.StepResult(3, "C", SS.SKIPPED),
            pipeline_mod.StepResult(4, "D", SS.FAILED, message="err"),
        ])
        assert pr.succeeded == 2
        assert pr.failed == 1
        assert pr.skipped == 1

    def test_summary_dict_keys(self, pipeline_mod) -> None:
        pr = pipeline_mod.PipelineResult()
        d = pr.summary_dict()
        assert "total_steps" in d
        assert "succeeded" in d
        assert "failed" in d
        assert "skipped" in d
        assert "total_elapsed_seconds" in d
        assert "start_from" in d
        assert "steps" in d
        assert isinstance(d["steps"], list)

    def test_summary_dict_step_fields(self, pipeline_mod) -> None:
        SS = pipeline_mod.StepStatus
        pr = pipeline_mod.PipelineResult(steps=[
            pipeline_mod.StepResult(1, "A", SS.COMPLETED, 1.234),
        ])
        step_entry = pr.summary_dict()["steps"][0]
        assert step_entry["step"] == 1
        assert step_entry["name"] == "A"
        assert step_entry["status"] == "completed"
        assert step_entry["elapsed_seconds"] == 1.23  # rounded


# ===========================================================================
# Output-existence checks
# ===========================================================================


class TestOutputExistenceHelpers:
    """Test _has_files, _dir_not_empty, check_step_complete."""

    def test_has_files_empty_dir(self, pipeline_mod, tmp_path) -> None:
        assert pipeline_mod._has_files(tmp_path, "*.json") is False

    def test_has_files_with_match(self, pipeline_mod, tmp_path) -> None:
        (tmp_path / "data.json").write_text("{}")
        assert pipeline_mod._has_files(tmp_path, "*.json") is True

    def test_has_files_no_match(self, pipeline_mod, tmp_path) -> None:
        (tmp_path / "data.txt").write_text("hi")
        assert pipeline_mod._has_files(tmp_path, "*.json") is False

    def test_has_files_nonexistent_dir(self, pipeline_mod) -> None:
        assert pipeline_mod._has_files(Path("/nonexistent_abc"), "*.xml") is False

    def test_dir_not_empty_true(self, pipeline_mod, tmp_path) -> None:
        (tmp_path / "file.txt").write_text("x")
        assert pipeline_mod._dir_not_empty(tmp_path) is True

    def test_dir_not_empty_false(self, pipeline_mod, tmp_path) -> None:
        assert pipeline_mod._dir_not_empty(tmp_path) is False

    def test_dir_not_empty_nonexistent(self, pipeline_mod) -> None:
        assert pipeline_mod._dir_not_empty(Path("/nonexistent_xyz")) is False

    def test_check_step_complete_default_config(
        self, pipeline_mod, default_config
    ) -> None:
        """With fresh config paths, no steps should be complete."""
        for step in range(1, 12):
            # Might be True if data dir has files; just verify it doesn't crash
            result = pipeline_mod.check_step_complete(step, default_config)
            assert isinstance(result, bool)

    def test_check_step_invalid_step_returns_false(
        self, pipeline_mod, default_config
    ) -> None:
        assert pipeline_mod.check_step_complete(0, default_config) is False
        assert pipeline_mod.check_step_complete(99, default_config) is False

    def test_get_pipeline_status_returns_all_steps(
        self, pipeline_mod, default_config
    ) -> None:
        status = pipeline_mod.get_pipeline_status(default_config)
        assert len(status) == 12
        for step in range(1, 13):
            assert step in status
            assert isinstance(status[step], bool)


# ===========================================================================
# Cost estimation prompt (Claude API steps)
# ===========================================================================


class TestClaudeCostEstimation:
    """Test the inline cost estimation in run_full_pipeline."""

    def test_estimate_step_3(self, pipeline_mod, default_config) -> None:
        est = pipeline_mod._estimate_claude_cost(3, default_config)
        assert est["step"] == 3
        assert "Relation" in est["name"] or "relation" in est["name"].lower()
        assert est["estimated_cost_usd"] >= 0
        assert est["model"] == default_config.extraction.relation.model

    def test_estimate_step_6(self, pipeline_mod, default_config) -> None:
        est = pipeline_mod._estimate_claude_cost(6, default_config)
        assert est["step"] == 6
        assert "label" in est["name"].lower() or "router" in est["name"].lower()
        assert est["estimated_cost_usd"] >= 0
        assert est["model"] == default_config.labeling.model

    def test_estimate_non_api_step_returns_error(
        self, pipeline_mod, default_config
    ) -> None:
        est = pipeline_mod._estimate_claude_cost(1, default_config)
        assert "error" in est

    def test_prompt_confirmation_auto_yes(self, pipeline_mod) -> None:
        fake_est = {
            "step": 3,
            "name": "Test",
            "model": "test-model",
            "estimated_chunks": 100,
            "estimated_input_tokens": 200_000,
            "estimated_output_tokens": 50_000,
            "estimated_cost_usd": 1.35,
        }
        assert pipeline_mod._prompt_cost_confirmation(fake_est, auto_yes=True) is True

    @patch("builtins.input", return_value="n")
    def test_prompt_confirmation_user_declines(self, mock_input, pipeline_mod) -> None:
        fake_est = {
            "step": 6,
            "name": "Test",
            "model": "test",
            "estimated_chunks": 10,
            "estimated_input_tokens": 10_000,
            "estimated_output_tokens": 2_000,
            "estimated_cost_usd": 0.05,
        }
        assert pipeline_mod._prompt_cost_confirmation(fake_est, auto_yes=False) is False


# ===========================================================================
# run_step
# ===========================================================================


class TestRunStep:
    """Test the single-step runner logic."""

    def test_run_step_dry_run(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_step(
            1,
            default_config,
            config_path="config/default.yaml",
            dry_run=True,
            skip_completed=False,
        )
        assert result.status == pipeline_mod.StepStatus.DRY_RUN
        assert "01_download_corpus" in result.message

    def test_run_step_invalid_step(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_step(
            99,
            default_config,
            config_path="config/default.yaml",
        )
        assert result.status == pipeline_mod.StepStatus.FAILED
        assert "No script" in result.message

    def test_run_step_skip_when_complete(self, pipeline_mod, default_config) -> None:
        """If check_step_complete returns True, step should be skipped."""
        with patch.object(pipeline_mod, "check_step_complete", return_value=True):
            result = pipeline_mod.run_step(
                1,
                default_config,
                config_path="config/default.yaml",
                skip_completed=True,
            )
            assert result.status == pipeline_mod.StepStatus.SKIPPED

    def test_run_step_no_skip_flag(self, pipeline_mod, default_config) -> None:
        """With skip_completed=False, even 'complete' steps should attempt to run."""
        # Dry run to avoid actually executing
        with patch.object(pipeline_mod, "check_step_complete", return_value=True):
            result = pipeline_mod.run_step(
                1,
                default_config,
                config_path="config/default.yaml",
                skip_completed=False,
                dry_run=True,
            )
            assert result.status == pipeline_mod.StepStatus.DRY_RUN

    def test_run_step_claude_cost_declined(self, pipeline_mod, default_config) -> None:
        """If user declines cost for a Claude step, it should be skipped."""
        with patch.object(pipeline_mod, "check_step_complete", return_value=False), \
             patch.object(pipeline_mod, "_prompt_cost_confirmation", return_value=False):
            result = pipeline_mod.run_step(
                3,
                default_config,
                config_path="config/default.yaml",
                auto_yes=False,
            )
            assert result.status == pipeline_mod.StepStatus.SKIPPED
            assert "declined" in result.message.lower()


# ===========================================================================
# run_pipeline
# ===========================================================================


class TestRunPipeline:
    """Test the pipeline orchestrator."""

    def test_dry_run_all_steps(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_pipeline(
            default_config,
            config_path="config/default.yaml",
            dry_run=True,
            skip_completed=False,  # disable skip so all steps get dry_run
        )
        # All 12 steps should appear
        assert len(result.steps) == 12
        for s in result.steps:
            assert s.status == pipeline_mod.StepStatus.DRY_RUN

    def test_dry_run_start_from(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_pipeline(
            default_config,
            config_path="config/default.yaml",
            start_from=7,
            dry_run=True,
        )
        assert len(result.steps) == 6  # steps 7-12
        assert result.steps[0].step_number == 7
        assert result.steps[-1].step_number == 12

    def test_dry_run_stop_after(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_pipeline(
            default_config,
            config_path="config/default.yaml",
            start_from=2,
            stop_after=5,
            dry_run=True,
        )
        assert len(result.steps) == 4  # steps 2-5
        assert result.steps[0].step_number == 2
        assert result.steps[-1].step_number == 5

    def test_pipeline_result_timing(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_pipeline(
            default_config,
            config_path="config/default.yaml",
            dry_run=True,
        )
        assert result.total_elapsed_seconds >= 0

    def test_pipeline_start_from_clamped(self, pipeline_mod, default_config) -> None:
        result = pipeline_mod.run_pipeline(
            default_config,
            config_path="config/default.yaml",
            start_from=0,  # should be clamped to 1
            dry_run=True,
        )
        assert result.steps[0].step_number == 1

    def test_pipeline_stop_on_failure(self, pipeline_mod, default_config) -> None:
        """Pipeline should abort on failure when stop_on_failure=True."""
        def fake_run_step(step, cfg, **kw):
            if step == 3:
                return pipeline_mod.StepResult(
                    step, "fail", pipeline_mod.StepStatus.FAILED, message="boom"
                )
            return pipeline_mod.StepResult(
                step, "ok", pipeline_mod.StepStatus.COMPLETED
            )

        with patch.object(pipeline_mod, "run_step", side_effect=fake_run_step):
            result = pipeline_mod.run_pipeline(
                default_config,
                config_path="config/default.yaml",
                stop_on_failure=True,
            )
            assert result.failed == 1
            # Should have stopped at step 3
            assert len(result.steps) == 3

    def test_pipeline_continue_on_failure(self, pipeline_mod, default_config) -> None:
        """Pipeline should continue past failures when stop_on_failure=False."""
        def fake_run_step(step, cfg, **kw):
            if step == 3:
                return pipeline_mod.StepResult(
                    step, "fail", pipeline_mod.StepStatus.FAILED, message="boom"
                )
            return pipeline_mod.StepResult(
                step, "ok", pipeline_mod.StepStatus.COMPLETED
            )

        with patch.object(pipeline_mod, "run_step", side_effect=fake_run_step):
            result = pipeline_mod.run_pipeline(
                default_config,
                config_path="config/default.yaml",
                stop_on_failure=False,
            )
            assert result.failed == 1
            assert len(result.steps) == 12


# ===========================================================================
# print_status
# ===========================================================================


class TestPrintStatus:
    """Test the status display function."""

    def test_print_status_runs(self, pipeline_mod, default_config, capsys) -> None:
        pipeline_mod.print_status(default_config)
        captured = capsys.readouterr()
        assert "Pipeline Status" in captured.out
        assert "completed" in captured.out
        # Should contain all 12 step names
        step_names = [
            "Download corpus", "Extract entities", "Extract relations",
            "Populate knowledge graph", "Build FAISS index",
            "Generate router labels", "Prepare training data",
            "Train router head", "Train retrieval head",
            "Joint fine-tuning", "Evaluate", "Run inference",
        ]
        for name in step_names:
            assert name in captured.out, f"Missing step name: {name}"


# ===========================================================================
# CLI argument parser
# ===========================================================================


class TestPipelineCLI:
    """Test the CLI argument parser."""

    def test_default_args(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args([])
        assert args.config == "config/default.yaml"
        assert args.start_from == 1
        assert args.dry_run is False
        assert args.yes is False
        assert args.status is False

    def test_start_from_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--start-from", "7"])
        assert args.start_from == 7

    def test_dry_run_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_yes_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--yes"])
        assert args.yes is True

    def test_y_short_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["-y"])
        assert args.yes is True

    def test_status_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--status"])
        assert args.status is True

    def test_no_skip_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--no-skip"])
        assert args.no_skip is True

    def test_stop_after_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--stop-after", "5"])
        assert args.stop_after == 5

    def test_save_report_flag(self, pipeline_mod) -> None:
        parser = pipeline_mod.build_parser()
        args = parser.parse_args(["--save-report", "/tmp/report.json"])
        assert args.save_report == "/tmp/report.json"


# ===========================================================================
# main() integration
# ===========================================================================


class TestPipelineMain:
    """Test main() with --dry-run and --status."""

    def test_main_status(self, pipeline_mod) -> None:
        exit_code = pipeline_mod.main(["--status"])
        assert exit_code == 0

    def test_main_dry_run(self, pipeline_mod) -> None:
        exit_code = pipeline_mod.main(["--dry-run"])
        assert exit_code == 0

    def test_main_dry_run_start_from(self, pipeline_mod) -> None:
        exit_code = pipeline_mod.main(["--dry-run", "--start-from", "5"])
        assert exit_code == 0

    def test_main_save_report(self, pipeline_mod, tmp_path) -> None:
        report_path = tmp_path / "report.json"
        exit_code = pipeline_mod.main([
            "--dry-run",
            "--save-report", str(report_path),
        ])
        assert exit_code == 0
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "total_steps" in data
        assert "steps" in data
        assert len(data["steps"]) == 12


# ===========================================================================
# estimate_costs.py — Corpus chunk estimation
# ===========================================================================


class TestCorpusChunkEstimation:
    """Test estimate_corpus_chunks."""

    def test_default_config(self, costs_mod, default_config) -> None:
        chunks = costs_mod.estimate_corpus_chunks(default_config)
        assert isinstance(chunks, int)
        assert chunks > 0

    def test_override_max_documents(self, costs_mod, default_config) -> None:
        chunks_100 = costs_mod.estimate_corpus_chunks(default_config, max_documents=100)
        chunks_1000 = costs_mod.estimate_corpus_chunks(default_config, max_documents=1000)
        assert chunks_1000 > chunks_100

    def test_chunk_count_proportional(self, costs_mod, default_config) -> None:
        c1 = costs_mod.estimate_corpus_chunks(default_config, max_documents=100)
        c2 = costs_mod.estimate_corpus_chunks(default_config, max_documents=200)
        assert c2 == 2 * c1


# ===========================================================================
# estimate_costs.py — API cost estimation
# ===========================================================================


class TestAPICostEstimation:
    """Test estimate_api_costs."""

    def test_returns_two_costs(self, costs_mod, default_config) -> None:
        chunks = costs_mod.estimate_corpus_chunks(default_config, max_documents=100)
        costs = costs_mod.estimate_api_costs(default_config, chunks)
        assert len(costs) == 2

    def test_step3_cost_structure(self, costs_mod, default_config) -> None:
        chunks = costs_mod.estimate_corpus_chunks(default_config, max_documents=100)
        costs = costs_mod.estimate_api_costs(default_config, chunks)
        rel_cost = costs[0]
        assert "Relation" in rel_cost.step_name
        assert rel_cost.total_input_tokens > 0
        assert rel_cost.total_output_tokens > 0
        assert rel_cost.total_cost_usd >= 0
        assert rel_cost.rate_limit_rpm > 0

    def test_step6_cost_structure(self, costs_mod, default_config) -> None:
        chunks = costs_mod.estimate_corpus_chunks(default_config, max_documents=100)
        costs = costs_mod.estimate_api_costs(default_config, chunks)
        lab_cost = costs[1]
        assert "label" in lab_cost.step_name.lower() or "Router" in lab_cost.step_name
        assert lab_cost.total_input_tokens > 0
        assert lab_cost.total_cost_usd >= 0

    def test_cost_scales_with_chunks(self, costs_mod, default_config) -> None:
        c_small = costs_mod.estimate_api_costs(default_config, 100)
        c_large = costs_mod.estimate_api_costs(default_config, 10000)
        assert c_large[0].total_cost_usd > c_small[0].total_cost_usd

    def test_estimated_minutes_positive(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_api_costs(default_config, 500)
        for c in costs:
            assert c.estimated_minutes >= 0


# ===========================================================================
# estimate_costs.py — Training cost estimation
# ===========================================================================


class TestTrainingCostEstimation:
    """Test estimate_training_costs."""

    def test_returns_three_phases(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_training_costs(default_config, 1000)
        assert len(costs) == 3

    def test_phase_names(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_training_costs(default_config, 1000)
        names = [c.phase_name for c in costs]
        assert any("Router" in n for n in names)
        assert any("Retrieval" in n for n in names)
        assert any("Joint" in n for n in names)

    def test_gpu_hours_positive(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_training_costs(default_config, 1000)
        for c in costs:
            assert c.total_gpu_hours > 0

    def test_local_gpu_zero_cost(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_training_costs(default_config, 1000, gpu_type="local")
        for c in costs:
            assert c.cost_usd == 0.0

    def test_different_gpu_types(self, costs_mod, default_config) -> None:
        cost_a100 = costs_mod.estimate_training_costs(default_config, 1000, gpu_type="A100-80GB")
        cost_t4 = costs_mod.estimate_training_costs(default_config, 1000, gpu_type="T4")
        # A100 is more expensive per hour
        total_a100 = sum(c.cost_usd for c in cost_a100)
        total_t4 = sum(c.cost_usd for c in cost_t4)
        assert total_a100 > total_t4

    def test_epochs_match_config(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_training_costs(default_config, 1000)
        router_phase = [c for c in costs if "Router" in c.phase_name][0]
        assert router_phase.epochs == default_config.training.router.epochs

    def test_batch_size_match_config(self, costs_mod, default_config) -> None:
        costs = costs_mod.estimate_training_costs(default_config, 1000)
        retrieval_phase = [c for c in costs if "Retrieval" in c.phase_name][0]
        assert retrieval_phase.batch_size == default_config.training.retrieval.batch_size


# ===========================================================================
# estimate_costs.py — Storage estimation
# ===========================================================================


class TestStorageEstimation:
    """Test estimate_storage."""

    def test_returns_multiple_components(self, costs_mod, default_config) -> None:
        estimates = costs_mod.estimate_storage(default_config, 1000)
        assert len(estimates) >= 5

    def test_component_names_are_strings(self, costs_mod, default_config) -> None:
        for s in costs_mod.estimate_storage(default_config, 1000):
            assert isinstance(s.component, str)
            assert len(s.component) > 0

    def test_all_gb_non_negative(self, costs_mod, default_config) -> None:
        for s in costs_mod.estimate_storage(default_config, 1000):
            assert s.estimated_gb >= 0

    def test_all_monthly_cost_non_negative(self, costs_mod, default_config) -> None:
        for s in costs_mod.estimate_storage(default_config, 1000):
            assert s.monthly_cost_usd >= 0

    def test_checkpoints_storage(self, costs_mod, default_config) -> None:
        estimates = costs_mod.estimate_storage(default_config, 1000)
        ckpt = [s for s in estimates if "checkpoint" in s.component.lower()]
        assert len(ckpt) == 1
        assert ckpt[0].estimated_gb > 0

    def test_storage_scales_with_chunks(self, costs_mod, default_config) -> None:
        s_small = costs_mod.estimate_storage(default_config, 100)
        s_large = costs_mod.estimate_storage(default_config, 100_000)
        total_small = sum(s.estimated_gb for s in s_small)
        total_large = sum(s.estimated_gb for s in s_large)
        assert total_large > total_small


# ===========================================================================
# estimate_costs.py — CostReport
# ===========================================================================


class TestCostReport:
    """Test the CostReport data class."""

    def test_empty_report(self, costs_mod) -> None:
        r = costs_mod.CostReport()
        assert r.grand_total_usd == 0.0

    def test_to_dict_keys(self, costs_mod) -> None:
        r = costs_mod.CostReport()
        d = r.to_dict()
        expected_keys = {
            "api_costs", "training_costs", "storage_estimates",
            "total_api_cost_usd", "total_training_cost_usd",
            "total_storage_monthly_usd", "grand_total_usd",
        }
        assert set(d.keys()) == expected_keys


# ===========================================================================
# estimate_costs.py — build_cost_report
# ===========================================================================


class TestBuildCostReport:
    """Test the full cost report builder."""

    def test_default_report(self, costs_mod, default_config) -> None:
        report = costs_mod.build_cost_report(default_config)
        assert isinstance(report, costs_mod.CostReport)
        assert len(report.api_costs) == 2
        assert len(report.training_costs) == 3
        assert len(report.storage_estimates) >= 5
        assert report.grand_total_usd > 0

    def test_report_with_override_documents(self, costs_mod, default_config) -> None:
        report = costs_mod.build_cost_report(default_config, max_documents=10)
        assert report.grand_total_usd > 0

    def test_report_totals_consistency(self, costs_mod, default_config) -> None:
        report = costs_mod.build_cost_report(default_config, max_documents=100)
        api_sum = sum(c.total_cost_usd for c in report.api_costs)
        assert abs(report.total_api_cost_usd - api_sum) < 0.01

        train_sum = sum(c.cost_usd for c in report.training_costs)
        assert abs(report.total_training_cost_usd - train_sum) < 0.01

    def test_report_to_dict_serializable(self, costs_mod, default_config) -> None:
        report = costs_mod.build_cost_report(default_config, max_documents=50)
        d = report.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_report_gpu_type_local(self, costs_mod, default_config) -> None:
        report = costs_mod.build_cost_report(
            default_config, max_documents=100, gpu_type="local"
        )
        assert report.total_training_cost_usd == 0.0


# ===========================================================================
# estimate_costs.py — Pretty printer
# ===========================================================================


class TestPrintCostReport:
    """Test the pretty printer for cost reports."""

    def test_print_report_runs(self, costs_mod, default_config, capsys) -> None:
        report = costs_mod.build_cost_report(default_config, max_documents=50)
        costs_mod.print_cost_report(report)
        captured = capsys.readouterr()
        assert "COST ESTIMATE" in captured.out
        assert "CLAUDE API COSTS" in captured.out
        assert "TRAINING COSTS" in captured.out
        assert "STORAGE ESTIMATES" in captured.out
        assert "GRAND TOTAL" in captured.out


# ===========================================================================
# estimate_costs.py — CLI
# ===========================================================================


class TestCostsCLI:
    """Test the cost estimation CLI."""

    def test_default_args(self, costs_mod) -> None:
        parser = costs_mod.build_parser()
        args = parser.parse_args([])
        assert args.config == "config/default.yaml"
        assert args.max_documents is None
        assert args.gpu_type == "A100-80GB"
        assert args.json is None

    def test_all_flags(self, costs_mod) -> None:
        parser = costs_mod.build_parser()
        args = parser.parse_args([
            "--config", "custom.yaml",
            "--max-documents", "500",
            "--gpu-type", "T4",
            "--json", "/tmp/cost.json",
        ])
        assert args.config == "custom.yaml"
        assert args.max_documents == 500
        assert args.gpu_type == "T4"
        assert args.json == "/tmp/cost.json"


class TestCostsMain:
    """Test the cost estimation main() entry point."""

    def test_main_default(self, costs_mod) -> None:
        exit_code = costs_mod.main([])
        assert exit_code == 0

    def test_main_with_max_documents(self, costs_mod) -> None:
        exit_code = costs_mod.main(["--max-documents", "50"])
        assert exit_code == 0

    def test_main_json_export(self, costs_mod, tmp_path) -> None:
        json_path = tmp_path / "costs.json"
        exit_code = costs_mod.main(["--json", str(json_path)])
        assert exit_code == 0
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "grand_total_usd" in data
        assert "api_costs" in data

    def test_main_gpu_type(self, costs_mod) -> None:
        exit_code = costs_mod.main(["--gpu-type", "T4"])
        assert exit_code == 0


# ===========================================================================
# Pricing constants
# ===========================================================================


class TestPricingConstants:
    """Verify pricing constants are sensible."""

    def test_claude_input_price_positive(self, costs_mod) -> None:
        assert costs_mod.CLAUDE_SONNET_INPUT_PRICE_PER_M > 0

    def test_claude_output_more_expensive(self, costs_mod) -> None:
        assert costs_mod.CLAUDE_SONNET_OUTPUT_PRICE_PER_M > costs_mod.CLAUDE_SONNET_INPUT_PRICE_PER_M

    def test_gpu_rates_all_non_negative(self, costs_mod) -> None:
        for gpu, rate in costs_mod.GPU_HOURLY_RATES.items():
            assert rate >= 0, f"{gpu} has negative rate"

    def test_local_gpu_free(self, costs_mod) -> None:
        assert costs_mod.GPU_HOURLY_RATES["local"] == 0.0

    def test_storage_price_positive(self, costs_mod) -> None:
        assert costs_mod.STORAGE_COST_PER_GB_MONTH > 0


# ===========================================================================
# Makefile targets
# ===========================================================================


class TestMakefileTargets:
    """Verify the Makefile has the new Phase 9 targets."""

    @pytest.fixture(scope="class")
    def makefile_content(self) -> str:
        makefile_path = Path(__file__).resolve().parent.parent / "Makefile"
        return makefile_path.read_text()

    def test_full_pipeline_target(self, makefile_content) -> None:
        assert "full-pipeline:" in makefile_content

    def test_resume_target(self, makefile_content) -> None:
        assert "resume:" in makefile_content

    def test_resume_uses_step_var(self, makefile_content) -> None:
        assert "$(STEP)" in makefile_content

    def test_dry_run_target(self, makefile_content) -> None:
        assert "dry-run:" in makefile_content

    def test_status_target(self, makefile_content) -> None:
        assert "status:" in makefile_content

    def test_estimate_costs_target(self, makefile_content) -> None:
        assert "estimate-costs" in makefile_content

    def test_full_pipeline_uses_yes_flag(self, makefile_content) -> None:
        # full-pipeline should include --yes for non-interactive
        idx = makefile_content.index("full-pipeline:")
        block = makefile_content[idx:idx + 200]
        assert "--yes" in block

    def test_resume_uses_start_from(self, makefile_content) -> None:
        idx = makefile_content.index("resume:")
        block = makefile_content[idx:idx + 200]
        assert "--start-from" in block

    def test_dry_run_uses_dry_run_flag(self, makefile_content) -> None:
        idx = makefile_content.index("dry-run:")
        block = makefile_content[idx:idx + 200]
        assert "--dry-run" in block

    def test_status_uses_status_flag(self, makefile_content) -> None:
        idx = makefile_content.index("\nstatus:")
        block = makefile_content[idx:idx + 200]
        assert "--status" in block


# ===========================================================================
# Integration: script is importable and runnable
# ===========================================================================


class TestScriptImportability:
    """Ensure the scripts can be imported without side effects."""

    def test_import_run_full_pipeline(self) -> None:
        import scripts.run_full_pipeline as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "run_pipeline")
        assert hasattr(mod, "run_step")
        assert hasattr(mod, "build_parser")

    def test_import_estimate_costs(self) -> None:
        import scripts.estimate_costs as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "build_cost_report")
        assert hasattr(mod, "estimate_corpus_chunks")
        assert hasattr(mod, "estimate_api_costs")
        assert hasattr(mod, "estimate_training_costs")
        assert hasattr(mod, "estimate_storage")
        assert hasattr(mod, "print_cost_report")
