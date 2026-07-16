"""Shared pytest fixtures."""

import pathlib

import pytest

from curios_comic_crawler.config import AppConfig


@pytest.fixture
def valid_config_dict() -> dict:
    """A minimal `config.json`-shaped dict that passes `AppConfig` validation."""
    return {
        'root_site': 'https://www.collectedcurios.com/',
        'BD_name': 'SA',
        'padded': 4,
        'fails': 2,
        'filename_variants': ['small', 'small_b'],
        'ext': '.jpg',
        'folder_data': 'data/',
        'folder_save_dl': 'DL/',
        'folder_save_upscale': 'UPSCALE/',
        'folder_models': 'models/',
        'upscaler': {'engine': 'opencv', 'model_name': 'edsr', 'model_scale': 3},
        'gray_values': 32,
        'headers': {'User-Agent': 'Mozilla/5.0'},
    }


@pytest.fixture
def make_config(valid_config_dict: dict):
    """Factory returning a valid `AppConfig`, optionally overridden field-by-field."""

    def _make(**overrides: object) -> AppConfig:
        return AppConfig.model_validate({**valid_config_dict, **overrides})

    return _make


@pytest.fixture
def config(make_config, tmp_path: pathlib.Path) -> AppConfig:
    """A valid `AppConfig` rooted at a fresh `tmp_path`, with its data folders pre-created."""
    app_config = make_config(folder_data=str(tmp_path / 'data') + '/')
    app_config.dl_dir.mkdir(parents=True, exist_ok=True)
    app_config.upscale_dir.mkdir(parents=True, exist_ok=True)
    app_config.models_dir.mkdir(parents=True, exist_ok=True)
    return app_config
