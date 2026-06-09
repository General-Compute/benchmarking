from pathlib import Path

import pytest

from benchmarking.config.settings import Settings


def test_settings_uses_packaged_default_config_outside_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("RESULTS_DIR", raising=False)

    settings = Settings()

    providers = settings.get_all_provider_configs()
    workloads = settings.get_all_workload_configs()

    assert set(providers) == {"general_compute", "openrouter"}
    assert "ctx_256" in workloads


def test_config_file_env_override_missing_file_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setenv("CONFIG_FILE", str(missing))

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Settings().load_yaml_config()


def test_results_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "custom-results"
    monkeypatch.setenv("RESULTS_DIR", str(results_dir))

    assert Settings().get_output_config().results_dir == str(results_dir)
