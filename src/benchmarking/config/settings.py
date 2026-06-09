"""Configuration management for benchmarking tool."""

import os
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, cast

import yaml
from dotenv import load_dotenv

from .models import (
    BenchmarkConfig,
    GenerationParams,
    ModelConfig,
    OutputConfig,
    ProviderConfig,
    WorkloadConfig,
)


class Settings:
    """Application settings backed by config.yaml + environment variables.

    API keys are resolved dynamically from the env var named by each provider's
    `api_key_env`, so adding a provider requires no code change here.
    """

    def __init__(self, config_file: str | None = None):
        load_dotenv()
        self.config_file = config_file or os.getenv("CONFIG_FILE")
        self.results_dir_override = os.getenv("RESULTS_DIR")
        self._config_cache: Dict[str, Any] | None = None

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load (and cache) the YAML configuration file."""
        if self._config_cache is None:
            config = self._read_yaml_config()
            if not isinstance(config, dict):
                raise ValueError("Config file must contain a YAML mapping")
            self._config_cache = cast(Dict[str, Any], config)
        return self._config_cache

    def _read_yaml_config(self) -> Any:
        """Read explicit, local, or packaged default configuration."""
        if self.config_file:
            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            with config_path.open() as f:
                return yaml.safe_load(f)

        local_config = Path("config/config.yaml")
        if local_config.exists():
            with local_config.open() as f:
                return yaml.safe_load(f)

        default_config = files("benchmarking.config").joinpath("default_config.yaml")
        with default_config.open() as f:
            return yaml.safe_load(f)

    # ---- Providers ----------------------------------------------------------

    def get_all_provider_configs(self) -> Dict[str, ProviderConfig]:
        config = self.load_yaml_config()
        return {name: ProviderConfig(**data) for name, data in config["providers"].items()}

    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        providers = self.get_all_provider_configs()
        if provider_name not in providers:
            raise KeyError(f"Unknown provider: {provider_name}")
        return providers[provider_name]

    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        return {name: cfg for name, cfg in self.get_all_provider_configs().items() if cfg.enabled}

    def get_api_key(self, provider_name: str) -> str:
        """Resolve a provider's API key from its configured env var."""
        provider = self.get_provider_config(provider_name)
        key = os.getenv(provider.api_key_env)
        if not key:
            raise ValueError(
                f"Missing API key for provider '{provider_name}': "
                f"set {provider.api_key_env} in your environment / .env"
            )
        return key

    def get_base_url(self, provider_name: str) -> str:
        """Resolve a provider's base URL.

        Uses the env var named by `base_url_env` when set, otherwise the
        literal `base_url` from config.
        """
        provider = self.get_provider_config(provider_name)
        if provider.base_url_env:
            url = os.getenv(provider.base_url_env)
            if not url:
                raise ValueError(
                    f"Missing base URL for provider '{provider_name}': "
                    f"set {provider.base_url_env} in your environment / .env"
                )
            return url
        if not provider.base_url:
            raise ValueError(
                f"Provider '{provider_name}' has no base_url or base_url_env configured"
            )
        return provider.base_url

    # ---- Models -------------------------------------------------------------

    def get_all_model_configs(self) -> Dict[str, ModelConfig]:
        config = self.load_yaml_config()
        return {name: ModelConfig(**data) for name, data in config["models"].items()}

    # ---- Workloads ----------------------------------------------------------

    def get_workload_config(self, workload_name: str) -> WorkloadConfig:
        config = self.load_yaml_config()
        return WorkloadConfig(**config["workloads"][workload_name])

    def get_all_workload_configs(self) -> Dict[str, WorkloadConfig]:
        config = self.load_yaml_config()
        return {name: WorkloadConfig(**data) for name, data in config["workloads"].items()}

    # ---- Misc ---------------------------------------------------------------

    def get_generation_params(self) -> GenerationParams:
        config = self.load_yaml_config()
        return GenerationParams(**config["generation_params"])

    def get_benchmark_config(self) -> BenchmarkConfig:
        config = self.load_yaml_config()
        return BenchmarkConfig(**config["benchmark"])

    def get_output_config(self) -> OutputConfig:
        config = self.load_yaml_config()
        output_data = dict(config.get("output", {}))
        if self.results_dir_override:
            output_data["results_dir"] = self.results_dir_override
        return OutputConfig(**output_data)
