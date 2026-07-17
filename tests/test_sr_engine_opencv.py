"""Tests for `curios_comic_crawler.sr_engine_opencv`.

`OpenCVEngine.__init__`/`upscale` need a real (large, downloaded) OpenCV `.pb` model to load via
`cv2.dnn_superres`, so they aren't exercised here -- covered by manual end-to-end runs instead.
`prepare()`'s composition logic (resolving the right model file/algorithm/scale for a given
config) is cheap to test without a real download, so it's covered here by faking out
`ensure_model`; this module previously had no test coverage at all.
"""

import pathlib

import pytest

from curios_comic_crawler import sr_engine_opencv
from curios_comic_crawler.config import OpenCVUpscaleConfig


def test_prepare_resolves_model_path_and_manifest_spec(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / 'EDSR_x3.pb'
    monkeypatch.setattr(sr_engine_opencv, 'ensure_model', lambda *_a, **_k: model_path)

    upscaler_config = OpenCVUpscaleConfig(model_name='edsr', model_scale=3)
    engine_init = sr_engine_opencv.prepare(upscaler_config, tmp_path)

    assert engine_init.model_path == model_path
    assert engine_init.algorithm == 'edsr'
    assert engine_init.scale == 3


def test_prepare_passes_models_dir_and_config_through_to_ensure_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_ensure_model(models_dir: pathlib.Path, model_name: str, model_scale: int) -> pathlib.Path:
        captured['models_dir'] = models_dir
        captured['model_name'] = model_name
        captured['model_scale'] = model_scale
        return models_dir / 'FSRCNN_x2.pb'

    monkeypatch.setattr(sr_engine_opencv, 'ensure_model', _fake_ensure_model)

    upscaler_config = OpenCVUpscaleConfig(model_name='fsrcnn', model_scale=2)
    engine_init = sr_engine_opencv.prepare(upscaler_config, tmp_path)

    assert captured == {'models_dir': tmp_path, 'model_name': 'fsrcnn', 'model_scale': 2}
    assert engine_init.model_path == tmp_path / 'FSRCNN_x2.pb'
    assert engine_init.algorithm == 'fsrcnn'
