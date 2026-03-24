"""Tests for YAML experiment config loading and metadata tracking."""

import json

import pytest

from projects.ml.cli.experiment_config import ExperimentConfig, load_experiment_config
from projects.ml.cli.metadata import build_metadata, load_metadata, save_metadata


class TestExperimentConfig:
    def test_load_yaml_config(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "name: test_experiment\n"
            "mode: model_comparison\n"
            "models: [ridge, xgboost]\n"
            "features: [har]\n"
            "train_window: 500\n"
            "horizon: 1\n"
            "total_chunks: 50\n"
            "backend: slurm\n"
        )
        cfg = load_experiment_config(config_file)
        assert cfg.name == "test_experiment"
        assert cfg.mode == "model_comparison"
        assert cfg.models == ["ridge", "xgboost"]
        assert cfg.features == ["har"]
        assert cfg.train_window == 500
        assert cfg.total_chunks == 50

    def test_load_json_config(self, tmp_path):
        config_file = tmp_path / "test.json"
        config_file.write_text(
            json.dumps(
                {
                    "name": "json_test",
                    "mode": "subgroup_analysis",
                    "models": ["ridge"],
                    "subgroups": ["baseline", "moments"],
                }
            )
        )
        cfg = load_experiment_config(config_file)
        assert cfg.name == "json_test"
        assert cfg.mode == "subgroup_analysis"
        assert cfg.subgroups == ["baseline", "moments"]

    def test_missing_mode_raises(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("name: no_mode\nmodels: [ridge]\n")
        with pytest.raises(ValueError, match="mode"):
            load_experiment_config(config_file)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_experiment_config("/nonexistent/path.yaml")

    def test_to_dict(self):
        cfg = ExperimentConfig(name="test", mode="naive")
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["mode"] == "naive"
        assert isinstance(d["models"], list)

    def test_defaults(self, tmp_path):
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text("mode: naive\n")
        cfg = load_experiment_config(config_file)
        assert cfg.name == "minimal"  # inferred from filename
        assert cfg.train_window == 500
        assert cfg.horizon == 1
        assert cfg.backend == "slurm"

    def test_comments_and_quotes(self, tmp_path):
        config_file = tmp_path / "commented.yaml"
        config_file.write_text(
            "# This is a comment\nname: 'quoted_name'\nmode: naive\nnotes: \"some notes here\"  # inline comment\n"
        )
        cfg = load_experiment_config(config_file)
        assert cfg.name == "quoted_name"
        assert cfg.notes == "some notes here"


class TestMetadata:
    def test_build_metadata(self):
        meta = build_metadata({"test_key": "test_value"})
        assert "timestamp" in meta
        assert "python_version" in meta
        assert "git_hash" in meta
        assert meta["experiment_config"]["test_key"] == "test_value"

    def test_save_and_load_metadata(self, tmp_path):
        meta = build_metadata({"model": "ridge"})
        save_metadata(tmp_path, meta)
        loaded = load_metadata(tmp_path)
        assert loaded is not None
        assert loaded["experiment_config"]["model"] == "ridge"

    def test_load_missing_metadata(self, tmp_path):
        assert load_metadata(tmp_path) is None


class TestDryRunBackend:
    def test_dry_run_backend_registered(self):
        from core.backends import get_backend

        backend = get_backend("dry-run")
        assert backend is not None

    def test_dry_run_prints(self, capsys):
        from core.backends import get_backend

        backend = get_backend("dry-run")
        backend.submit_array(
            "test_job",
            100,
            50,
            {"RESULT_DIR": "/tmp/test", "MODEL_TYPE": "ridge", "TOTAL_CHUNKS": "100", "EXTRA_ARGS": ""},
        )
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "ridge" in captured.out
