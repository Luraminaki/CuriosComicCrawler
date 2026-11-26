"""Shared pytest fixtures."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from curios_comic_crawler.config import AppConfig

if TYPE_CHECKING:
    import pathlib

ConfigFactory = Callable[..., AppConfig]
"""What the `make_config` fixture returns: keyword overrides in, a validated config out."""


@pytest.fixture(name="valid_config_dict")
def fixture_valid_config_dict() -> dict:
    """A minimal `config.json`-shaped dict that passes `AppConfig` validation."""
    return {
        "root_site": "https://www.collectedcurios.com/",
        "BD_name": "SA",
        "padded": 4,
        "fails": 2,
        "filename_variants": ["small", "small_b"],
        "ext": ".jpg",
        "folder_data": "data/",
        "folder_save_dl": "DL/",
        "folder_save_upscale": "UPSCALE/",
        "folder_models": "models/",
        "upscaler": {"engine": "opencv", "model_name": "edsr", "model_scale": 3},
        "gray_values": 32,
        "headers": {"User-Agent": "Mozilla/5.0"},
    }


@pytest.fixture(name="make_config")
def fixture_make_config(valid_config_dict: dict) -> ConfigFactory:
    """Factory returning a valid `AppConfig`, optionally overridden field-by-field."""

    def _make(**overrides: object) -> AppConfig:
        """Validate the base config with `overrides` applied.

        Args:
            **overrides: Config keys to replace.

        Returns:
            AppConfig: The validated config.
        """
        return AppConfig.model_validate({**valid_config_dict, **overrides})

    return _make


@pytest.fixture(name="config")
def fixture_config(make_config: ConfigFactory, tmp_path: "pathlib.Path") -> AppConfig:
    """A valid `AppConfig` rooted at a fresh `tmp_path`, with its data folders pre-created."""
    app_config = make_config(folder_data=str(tmp_path / "data") + "/")
    app_config.dl_dir.mkdir(parents=True, exist_ok=True)
    app_config.upscale_dir.mkdir(parents=True, exist_ok=True)
    app_config.models_dir.mkdir(parents=True, exist_ok=True)
    return app_config
