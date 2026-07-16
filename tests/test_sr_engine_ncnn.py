"""Tests for `curios_comic_crawler.sr_engine_ncnn`.

`realesrgan_ncnn_py` is an optional dependency (the `ncnn` extra), so these tests fake it out
via `sys.modules` rather than requiring it to be installed.
"""

import sys
import types
from typing import get_args

import numpy as np
import pytest

from curios_comic_crawler import sr_engine_ncnn
from curios_comic_crawler.config import NcnnUpscaleConfig
from curios_comic_crawler.models import NcnnModelName


class _FakeRealesrgan:
    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs

    def process_cv2(self, image: np.ndarray) -> np.ndarray:
        return image * 2


@pytest.fixture
def fake_realesrgan_ncnn_py(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType('realesrgan_ncnn_py')
    fake_module.Realesrgan = _FakeRealesrgan  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'realesrgan_ncnn_py', fake_module)


def test_every_ncnn_model_name_has_an_index(fake_realesrgan_ncnn_py: None) -> None:
    for model in get_args(NcnnModelName):
        sr_engine_ncnn.build(sr_engine_ncnn.NcnnEngineInit(model=model))  # must not raise KeyError


def test_engine_upscale_delegates_to_process_cv2(fake_realesrgan_ncnn_py: None) -> None:
    engine = sr_engine_ncnn.build(sr_engine_ncnn.NcnnEngineInit(model='realesrgan-x4plus-anime'))
    image = np.ones((2, 2, 3), dtype=np.uint8)

    result = engine.upscale(image)

    assert np.array_equal(result, image * 2)


def test_engine_runs_on_cpu_only(fake_realesrgan_ncnn_py: None) -> None:
    engine = sr_engine_ncnn.build(sr_engine_ncnn.NcnnEngineInit(model='realesrgan-x4plus'))

    assert engine._realesrgan.init_kwargs == {'gpuid': -1, 'model': 4}


def test_prepare_passes_through_configured_model() -> None:
    upscaler_config = NcnnUpscaleConfig(ncnn_model='realesr-animevideov3-x2')

    assert sr_engine_ncnn.prepare(upscaler_config) == sr_engine_ncnn.NcnnEngineInit(model='realesr-animevideov3-x2')


def test_missing_package_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'realesrgan_ncnn_py', None)  # simulates "not installed"

    with pytest.raises(RuntimeError, match='ncnn'):
        sr_engine_ncnn.build(sr_engine_ncnn.NcnnEngineInit(model='realesrgan-x4plus-anime'))
