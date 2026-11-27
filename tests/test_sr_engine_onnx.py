"""Tests for `curios_comic_crawler.sr_engine_onnx`.

`onnxruntime` is a base dependency and the models ship inside the package, so these run
against the real engine -- no mocking needed, unlike the ncnn engine this replaced.
"""

# pylint: disable=protected-access  # unit tests exercise the module's private helpers

from typing import get_args

import numpy as np
import pytest

from curios_comic_crawler import sr_engine_onnx
from curios_comic_crawler._http import sha256_of
from curios_comic_crawler.config import OnnxUpscaleConfig
from curios_comic_crawler.models import OnnxModelName

#: The default and lightest bundled model: keeps the real-engine tests fast.
LIGHT_MODEL: OnnxModelName = "realesr-animevideov3-x4"

# Guards against a bundled asset being silently corrupted or swapped for something else.
# If a regenerated model's hash legitimately changes, update it here to match.
_EXPECTED_SHA256: dict[OnnxModelName, str] = {
    "realesr-animevideov3-x4": "eb179c2a359dab4a4df3316663a0bcae879afc1674aaf4a012c11999fc554b7c",
    "realesrgan-x4plus-anime-6b": (
        "15c99fd2dd3a1c177267394e96b6d9bdaaaba1c0dffd92af9d8e94bf1de05df6"
    ),
    "realesr-general-x4v3": "32eeba5a4bd37287183c6f16d4f721ac3b893a80c14d0c4246d4e90fab46259a",
    "realesrgan-x4plus": "94cf5497dca9e68acb89df62f0fd6c375e6f32f32a1a3aa9f422a4777eb95ede",
}


def test_every_onnx_model_name_has_a_filename_and_expected_hash() -> None:
    """Every ONNX model offered has a bundled file and a known checksum."""
    for model in get_args(OnnxModelName):
        assert model in sr_engine_onnx._MODEL_FILENAMES
        assert model in _EXPECTED_SHA256


@pytest.mark.parametrize("model", get_args(OnnxModelName))
def test_bundled_model_matches_expected_hash(model: OnnxModelName) -> None:
    """The bundled ONNX models are byte-for-byte the expected ones."""
    model_path = sr_engine_onnx._ASSETS_DIR / sr_engine_onnx._MODEL_FILENAMES[model]
    assert sha256_of(model_path) == _EXPECTED_SHA256[model]  # streamed: the files are 64 MB


@pytest.mark.parametrize("model", get_args(OnnxModelName))
def test_prepare_resolves_bundled_model_path(model: OnnxModelName) -> None:
    """Preparing the ONNX engine finds the bundled model file."""
    engine_init = sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model=model))

    assert engine_init.model_path.is_file()
    assert engine_init.model_path.name == sr_engine_onnx._MODEL_FILENAMES[model]


def test_prepare_raises_if_asset_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing bundled model raises `FileNotFoundError`."""
    monkeypatch.setitem(sr_engine_onnx._MODEL_FILENAMES, LIGHT_MODEL, "does-not-exist.onnx")

    with pytest.raises(FileNotFoundError):
        _ = sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model=LIGHT_MODEL))


@pytest.mark.parametrize("model", get_args(OnnxModelName))
def test_build_and_upscale_produces_4x_output(model: OnnxModelName) -> None:
    """Each bundled model upscales an image by 4 in both dimensions."""
    engine_init = sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model=model))
    engine = sr_engine_onnx.OnnxEngine(engine_init, worker_count=1)

    image = np.random.default_rng(0).integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    result = engine.upscale(image)

    assert result.shape == (32, 48, 3)
    assert result.dtype == np.uint8


class _FixedOutputSession:
    """Stands in for an `InferenceSession`, returning a fixed model output."""

    def __init__(self, output: np.ndarray) -> None:
        """Set the output.

        Args:
            output: `(1, 3, H, W)` float32 RGB, as the model returns it.
        """
        self.output: np.ndarray = output

    def run(self, *_args: object) -> list[np.ndarray]:
        """Return the fixed output.

        Args:
            *_args: The output names and input feed (ignored).

        Returns:
            A one-element list, like ONNX Runtime.
        """
        return [self.output.copy()]


def _engine_returning(output: np.ndarray) -> sr_engine_onnx.OnnxEngine:
    """Build the light model's engine, then make it return `output` whatever the input.

    Args:
        output: The model output to return.

    Returns:
        The engine.
    """
    engine_init = sr_engine_onnx.prepare(OnnxUpscaleConfig(onnx_model=LIGHT_MODEL))
    engine = sr_engine_onnx.OnnxEngine(engine_init, worker_count=1)
    engine._session = _FixedOutputSession(output)  # pyright: ignore[reportAttributeAccessIssue]
    return engine


def test_output_is_converted_back_to_bgr() -> None:
    """The model's RGB planes come back in OpenCV's BGR order."""
    rgb = np.zeros((1, 3, 2, 2), dtype=np.float32)
    rgb[0, 2] = 1.0  # pure blue
    result = _engine_returning(rgb).upscale(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0].tolist() == [255, 0, 0]


def test_output_outside_the_unit_range_saturates() -> None:
    """Overshoot, however far, clamps to black or white instead of wrapping around."""
    values = np.array([-1e9, -0.3, 0.0, 0.5, 1.0, 1.2, 1e9], dtype=np.float32)
    output = np.broadcast_to(values, (1, 3, 1, values.size)).copy()
    result = _engine_returning(output).upscale(np.zeros((1, values.size, 3), dtype=np.uint8))

    # 0.5 * 255 is a rounding tie: 127 or 128 are both right.
    assert result[0, :, 0].tolist() in (
        [0, 0, 0, 128, 255, 255, 255],
        [0, 0, 0, 127, 255, 255, 255],
    )


def test_output_matches_the_reference_conversion_within_rounding() -> None:
    """Same pixels as the plain float conversion (scale, clip, round to nearest)."""
    output = np.random.default_rng(0).uniform(-0.2, 1.2, size=(1, 3, 16, 16)).astype(np.float32)
    reference = np.rint(np.clip(output[0] * 255.0, 0, 255)).astype(np.uint8).transpose(1, 2, 0)
    result = _engine_returning(output).upscale(np.zeros((16, 16, 3), dtype=np.uint8))

    assert np.array_equal(result[..., ::-1], reference)
