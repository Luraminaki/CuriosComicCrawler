"""Tests for `curios_comic_crawler.sr_engine_opencv`.

`OpenCVEngine.__init__`/`upscale` need a real (large, downloaded) OpenCV `.pb` model to load via
`cv2.dnn_superres`: they're tested only when a previous run already downloaded EDSR x3 (tests
never use the network). `prepare()`'s composition logic (resolving the right model
file/algorithm/scale for a given config) is tested without a download, by faking out
`ensure_model`.
"""

import pathlib

import numpy as np
import pytest

from curios_comic_crawler import sr_engine_opencv
from curios_comic_crawler.config import OpenCVUpscaleConfig


def test_prepare_resolves_model_path_and_manifest_spec(
    tmp_path: "pathlib.Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Preparing the OpenCV engine resolves the model file and its manifest entry."""
    model_path = tmp_path / "EDSR_x3.pb"
    monkeypatch.setattr(sr_engine_opencv, "ensure_model", lambda *_a, **_k: model_path)

    upscaler_config = OpenCVUpscaleConfig(model_name="edsr", model_scale=3)
    engine_init = sr_engine_opencv.prepare(upscaler_config, tmp_path)

    assert engine_init.model_path == model_path
    assert engine_init.algorithm == "edsr"
    assert engine_init.scale == 3


def test_prepare_passes_models_dir_and_config_through_to_ensure_model(
    tmp_path: "pathlib.Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """The models folder and chosen model reach the model download step."""
    captured: dict[str, object] = {}

    def _fake_ensure_model(
        models_dir: "pathlib.Path", model_name: str, model_scale: int
    ) -> "pathlib.Path":
        """Record the arguments instead of downloading.

        Returns:
            A fixed model path.
        """
        captured["models_dir"] = models_dir
        captured["model_name"] = model_name
        captured["model_scale"] = model_scale
        return models_dir / "FSRCNN_x2.pb"

    monkeypatch.setattr(sr_engine_opencv, "ensure_model", _fake_ensure_model)

    upscaler_config = OpenCVUpscaleConfig(model_name="fsrcnn", model_scale=2)
    engine_init = sr_engine_opencv.prepare(upscaler_config, tmp_path)

    assert captured == {"models_dir": tmp_path, "model_name": "fsrcnn", "model_scale": 2}
    assert engine_init.model_path == tmp_path / "FSRCNN_x2.pb"
    assert engine_init.algorithm == "fsrcnn"


_CACHED_MODELS = pathlib.Path(__file__).parents[1] / "data" / "models"


@pytest.mark.skipif(
    not (_CACHED_MODELS / "EDSR_x3.pb").is_file(),
    reason="needs EDSR_x3.pb downloaded by a previous run (tests never use the network)",
)
def test_real_model_loads_and_upscales() -> None:
    """The installed OpenCV can read the real model files (OpenCV 5 can't: pinned below 5)."""
    engine_init = sr_engine_opencv.prepare(
        OpenCVUpscaleConfig(engine="opencv", model_name="edsr", model_scale=3), _CACHED_MODELS
    )
    image = np.full((6, 4, 3), 128, dtype=np.uint8)

    result = sr_engine_opencv.OpenCVEngine(engine_init).upscale(image)

    assert result.shape == (18, 12, 3)
    assert abs(int(result.mean()) - 128) <= 2
