"""Tests for `curios_comic_crawler.config`."""

import json
import pathlib

import pytest
from pydantic import ValidationError

from curios_comic_crawler.config import (
    AppConfig,
    OnnxUpscaleConfig,
    OpenCVUpscaleConfig,
    load_config,
)


def test_valid_config_validates(valid_config_dict: dict) -> None:
    """A complete config loads, with the upscaler's own settings typed by engine."""
    config = AppConfig.model_validate(valid_config_dict)

    assert config.bd_name == "SA"
    assert isinstance(config.upscaler, OpenCVUpscaleConfig)
    assert config.upscaler.model_name == "edsr"


def test_missing_required_field_is_rejected(valid_config_dict: dict) -> None:
    """A config without `root_site` is refused."""
    del valid_config_dict["root_site"]

    with pytest.raises(ValidationError):
        _ = AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize(
    "bad_root_site",
    [
        "ftp://example.com/",
        "https://example.com",
        "httpfoo-not-a-url/",
        "https:///",
        "example.com/",
    ],
)
def test_root_site_must_be_http_with_trailing_slash(
    valid_config_dict: dict, bad_root_site: str
) -> None:
    """`root_site` must be an http(s) URL ending with a slash."""
    valid_config_dict["root_site"] = bad_root_site

    with pytest.raises(ValidationError):
        _ = AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize("bad_bd_name", ["sub/dir", "sub\\dir", "..", "", "CON", "com1", "lpt9"])
def test_bd_name_rejects_path_segments_and_reserved_names(
    valid_config_dict: dict, bad_bd_name: str
) -> None:
    """`BD_name` can't hold a path segment or a Windows reserved name."""
    valid_config_dict["BD_name"] = bad_bd_name

    with pytest.raises(ValidationError):
        _ = AppConfig.model_validate(valid_config_dict)


def test_filename_variants_requires_at_least_one_entry(valid_config_dict: dict) -> None:
    """At least one filename variant is required."""
    valid_config_dict["filename_variants"] = []

    with pytest.raises(ValidationError):
        _ = AppConfig.model_validate(valid_config_dict)


def test_unknown_model_scale_combo_is_rejected(valid_config_dict: dict) -> None:
    """An OpenCV model/scale pair with no known model is refused, naming the model."""
    valid_config_dict["upscaler"] = {
        "engine": "opencv",
        "model_name": "lapsrn",
        "model_scale": 3,
    }  # 2/4/8 only

    with pytest.raises(ValidationError, match="lapsrn"):
        _ = AppConfig.model_validate(valid_config_dict)


def test_onnx_upscaler_config_validates(valid_config_dict: dict) -> None:
    """The ONNX engine's config loads with its chosen model."""
    valid_config_dict["upscaler"] = {"engine": "onnx", "onnx_model": "realesr-animevideov3-x4"}

    config = AppConfig.model_validate(valid_config_dict)

    assert isinstance(config.upscaler, OnnxUpscaleConfig)
    assert config.upscaler.onnx_model == "realesr-animevideov3-x4"


def test_onnx_upscaler_config_defaults_model(valid_config_dict: dict) -> None:
    """Without a model, the ONNX engine uses `realesr-animevideov3-x4`."""
    valid_config_dict["upscaler"] = {"engine": "onnx"}

    config = AppConfig.model_validate(valid_config_dict)

    assert isinstance(config.upscaler, OnnxUpscaleConfig)
    assert config.upscaler.onnx_model == "realesr-animevideov3-x4"


def test_unknown_upscaler_engine_is_rejected(valid_config_dict: dict) -> None:
    """An engine other than `onnx`/`opencv` is refused."""
    valid_config_dict["upscaler"] = {"engine": "pytorch", "model_name": "edsr", "model_scale": 3}

    with pytest.raises(ValidationError):
        _ = AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize("gray_values", [1, 256])
def test_gray_values_out_of_bounds_is_rejected(valid_config_dict: dict, gray_values: int) -> None:
    """`gray_values` must stay within 2-255."""
    valid_config_dict["gray_values"] = gray_values

    with pytest.raises(ValidationError):
        _ = AppConfig.model_validate(valid_config_dict)


def test_derived_paths_join_correctly(valid_config_dict: dict) -> None:
    """Download, upscale and model folders are derived from the data folder and comic name."""
    config = AppConfig.model_validate(valid_config_dict)

    assert config.dl_dir == pathlib.Path("data/DL/SA")
    assert config.upscale_dir == pathlib.Path("data/UPSCALE/SA")
    assert config.models_dir == pathlib.Path("data/models")


def test_load_config_missing_file_raises(tmp_path: pathlib.Path) -> None:
    """A missing config file raises `FileNotFoundError`."""
    with pytest.raises(FileNotFoundError):
        _ = load_config(tmp_path / "does_not_exist.json")


def test_load_config_reads_and_validates(tmp_path: pathlib.Path, valid_config_dict: dict) -> None:
    """A config file on disk is read and validated."""
    config_path = tmp_path / "config.json"
    _ = config_path.write_text(json.dumps(valid_config_dict), encoding="utf-8")

    config = load_config(config_path)

    assert config.bd_name == "SA"


@pytest.mark.parametrize("bad_variant", ["../../escape", "sub/dir", "sub\\dir", ""])
def test_filename_variants_must_stay_inside_a_filename(
    valid_config_dict: dict, bad_variant: str
) -> None:
    """A variant with a path separator would save pages outside the download folder."""
    valid_config_dict["filename_variants"] = ["small", bad_variant]

    with pytest.raises(ValidationError, match="filename_variants"):
        _ = AppConfig.model_validate(valid_config_dict)


@pytest.mark.parametrize("root_site", ["http://example.com/", "https://example.com/comics/"])
def test_http_and_https_root_sites_are_accepted(valid_config_dict: dict, root_site: str) -> None:
    """Both schemes, with or without a path, are valid sites."""
    valid_config_dict["root_site"] = root_site

    assert AppConfig.model_validate(valid_config_dict).root_site == root_site
