"""Tests for `curios_comic_crawler.config`."""

import json
import pathlib

import pytest
from pydantic import ValidationError

from curios_comic_crawler.config import AppConfig, OnnxUpscaleConfig, OpenCVUpscaleConfig, load_config


def test_valid_config_validates(valid_config_dict: dict) -> None:
    config = AppConfig.model_validate(valid_config_dict)

    assert config.bd_name == 'SA'
    assert isinstance(config.upscaler, OpenCVUpscaleConfig)
    assert config.upscaler.model_name == 'edsr'


def test_missing_required_field_is_rejected(valid_config_dict: dict) -> None:
    del valid_config_dict['root_site']

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize('bad_root_site', ['ftp://example.com/', 'https://example.com'])
def test_root_site_must_be_http_with_trailing_slash(valid_config_dict: dict, bad_root_site: str) -> None:
    valid_config_dict['root_site'] = bad_root_site

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize('bad_bd_name', ['sub/dir', 'sub\\dir', '..', '', 'CON', 'com1', 'lpt9'])
def test_bd_name_rejects_path_segments_and_reserved_names(valid_config_dict: dict, bad_bd_name: str) -> None:
    valid_config_dict['BD_name'] = bad_bd_name

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_filename_variants_requires_at_least_one_entry(valid_config_dict: dict) -> None:
    valid_config_dict['filename_variants'] = []

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_unknown_model_scale_combo_is_rejected(valid_config_dict: dict) -> None:
    valid_config_dict['upscaler'] = {'engine': 'opencv', 'model_name': 'lapsrn', 'model_scale': 3}  # 2/4/8 only

    with pytest.raises(ValidationError, match='lapsrn'):
        AppConfig.model_validate(valid_config_dict)


def test_onnx_upscaler_config_validates(valid_config_dict: dict) -> None:
    valid_config_dict['upscaler'] = {'engine': 'onnx', 'onnx_model': 'realesr-animevideov3-x4'}

    config = AppConfig.model_validate(valid_config_dict)

    assert isinstance(config.upscaler, OnnxUpscaleConfig)
    assert config.upscaler.onnx_model == 'realesr-animevideov3-x4'


def test_onnx_upscaler_config_defaults_model(valid_config_dict: dict) -> None:
    valid_config_dict['upscaler'] = {'engine': 'onnx'}

    config = AppConfig.model_validate(valid_config_dict)

    assert isinstance(config.upscaler, OnnxUpscaleConfig)
    assert config.upscaler.onnx_model == 'realesr-animevideov3-x4'


def test_unknown_upscaler_engine_is_rejected(valid_config_dict: dict) -> None:
    valid_config_dict['upscaler'] = {'engine': 'pytorch', 'model_name': 'edsr', 'model_scale': 3}

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize('gray_values', [1, 256])
def test_gray_values_out_of_bounds_is_rejected(valid_config_dict: dict, gray_values: int) -> None:
    valid_config_dict['gray_values'] = gray_values

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_config_dict)


def test_derived_paths_join_correctly(valid_config_dict: dict) -> None:
    config = AppConfig.model_validate(valid_config_dict)

    assert config.dl_dir == pathlib.Path('data/DL/SA')
    assert config.upscale_dir == pathlib.Path('data/UPSCALE/SA')
    assert config.models_dir == pathlib.Path('data/models')


def test_load_config_missing_file_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / 'does_not_exist.json')


def test_load_config_reads_and_validates(tmp_path: pathlib.Path, valid_config_dict: dict) -> None:
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(valid_config_dict), encoding='utf-8')

    config = load_config(config_path)

    assert config.bd_name == 'SA'
