"""Tests for `curios_comic_crawler.sr_engine_onnx`.

`onnxruntime` is a base dependency and the models ship inside the package, so these run
against the real engine -- no mocking needed, unlike the ncnn engine this replaced.
"""

import hashlib
from typing import get_args

import numpy as np
import pytest

from curios_comic_crawler import sr_engine_onnx
from curios_comic_crawler.config import OnnxUpscaleConfig
from curios_comic_crawler.models import OnnxModelName

# Guards against a bundled asset being silently corrupted or swapped for something else.
# If a regenerated model's hash legitimately changes, update it here to match.
_EXPECTED_SHA256: dict[OnnxModelName, str] = {
    'realesr-animevideov3-x4': 'eb179c2a359dab4a4df3316663a0bcae879afc1674aaf4a012c11999fc554b7c',
    'realesrgan-x4plus-anime-6b': '15c99fd2dd3a1c177267394e96b6d9bdaaaba1c0dffd92af9d8e94bf1de05df6',
    'realesr-general-x4v3': '32eeba5a4bd37287183c6f16d4f721ac3b893a80c14d0c4246d4e90fab46259a',
    'realesrgan-x4plus': '94cf5497dca9e68acb89df62f0fd6c375e6f32f32a1a3aa9f422a4777eb95ede',
}


def test_every_onnx_model_name_has_a_filename_and_expected_hash() -> None:
    for model in get_args(OnnxModelName):
        assert model in sr_engine_onnx._MODEL_FILENAMES
        assert model in _EXPECTED_SHA256


@pytest.mark.parametrize('model', get_args(OnnxModelName))
def test_bundled_model_matches_expected_hash(model: OnnxModelName) -> None:
    model_path = sr_engine_onnx._ASSETS_DIR / sr_engine_onnx._MODEL_FILENAMES[model]
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

    assert digest == _EXPECTED_SHA256[model]


@pytest.mark.parametrize('model', get_args(OnnxModelName))
def test_prepare_resolves_bundled_model_path(model: OnnxModelName) -> None:
    engine_init = sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model=model))

    assert engine_init.model_path.is_file()
    assert engine_init.model_path.name == sr_engine_onnx._MODEL_FILENAMES[model]


def test_prepare_raises_if_asset_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sr_engine_onnx._MODEL_FILENAMES, 'realesr-animevideov3-x4', 'does-not-exist.onnx')

    with pytest.raises(FileNotFoundError):
        sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model='realesr-animevideov3-x4'))


@pytest.mark.parametrize('model', get_args(OnnxModelName))
def test_build_and_upscale_produces_4x_output(model: OnnxModelName) -> None:
    engine_init = sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model=model))
    engine = sr_engine_onnx.build(engine_init, worker_count=1)

    image = np.random.default_rng(0).integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    result = engine.upscale(image)

    assert result.shape == (32, 48, 3)
    assert result.dtype == np.uint8
