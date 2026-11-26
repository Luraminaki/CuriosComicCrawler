"""Tests for `curios_comic_crawler.sr_engine`: each config and init type reaches its engine."""

import pathlib
from typing import TYPE_CHECKING

from curios_comic_crawler import sr_engine, sr_engine_onnx, sr_engine_opencv

if TYPE_CHECKING:
    import pytest
    from conftest import ConfigFactory

    from curios_comic_crawler.config import AppConfig


def _record_calls(
    monkeypatch: "pytest.MonkeyPatch", module: object, name: str, result: object
) -> list[tuple[object, ...]]:
    """Replace `module.<name>` (a function or a class) with a stub that records its arguments.

    Args:
        monkeypatch: Replaces the attribute for this test only.
        module: The engine module holding it.
        name: The attribute to replace (`prepare`, or the engine class).
        result: What the stub returns (stands for the prepared data or the built engine).

    Returns:
        The arguments of each call, in order.
    """
    calls: list[tuple[object, ...]] = []

    def stub(*args: object) -> object:
        """Record the call.

        Args:
            *args: The call's arguments.

        Returns:
            The scripted result.
        """
        calls.append(args)
        return result

    monkeypatch.setattr(module, name, stub)
    return calls


def test_opencv_config_is_prepared_by_the_opencv_engine(
    config: "AppConfig", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """An `engine: "opencv"` config is prepared with its section and the models folder."""
    prepared = object()
    opencv_calls = _record_calls(monkeypatch, sr_engine_opencv, "prepare", prepared)
    onnx_calls = _record_calls(monkeypatch, sr_engine_onnx, "prepare", None)

    assert sr_engine.prepare_engine(config) is prepared
    assert opencv_calls == [(config.upscaler, config.models_dir)]
    assert onnx_calls == []


def test_onnx_config_is_prepared_by_the_onnx_engine(
    make_config: "ConfigFactory", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """An `engine: "onnx"` config is prepared with its section alone (nothing to download)."""
    config = make_config(upscaler={"engine": "onnx"})
    prepared = object()
    onnx_calls = _record_calls(monkeypatch, sr_engine_onnx, "prepare", prepared)
    opencv_calls = _record_calls(monkeypatch, sr_engine_opencv, "prepare", None)

    assert sr_engine.prepare_engine(config) is prepared
    assert onnx_calls == [(config.upscaler,)]
    assert opencv_calls == []


def test_opencv_init_builds_an_opencv_engine(monkeypatch: "pytest.MonkeyPatch") -> None:
    """OpenCV init data builds an `OpenCVEngine` (the worker count is not its concern)."""
    built = object()
    opencv_calls = _record_calls(monkeypatch, sr_engine_opencv, "OpenCVEngine", built)
    onnx_calls = _record_calls(monkeypatch, sr_engine_onnx, "OnnxEngine", None)
    engine_init = sr_engine_opencv.OpenCVEngineInit(pathlib.Path("m.pb"), "edsr", 3)

    assert sr_engine.build_engine(engine_init, worker_count=2) is built
    assert opencv_calls == [(engine_init,)]
    assert onnx_calls == []


def test_onnx_init_builds_an_onnx_engine_with_the_worker_count(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """ONNX init data builds an `OnnxEngine`, told how many workers share the CPU."""
    built = object()
    onnx_calls = _record_calls(monkeypatch, sr_engine_onnx, "OnnxEngine", built)
    opencv_calls = _record_calls(monkeypatch, sr_engine_opencv, "OpenCVEngine", None)
    engine_init = sr_engine_onnx.OnnxEngineInit(pathlib.Path("m.onnx"))

    assert sr_engine.build_engine(engine_init, worker_count=3) is built
    assert onnx_calls == [(engine_init, 3)]
    assert opencv_calls == []
