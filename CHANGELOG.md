# Changelog

All notable changes to this project are documented here. Versioning loosely
follows [Semantic Versioning](https://semver.org/).

## 0.5.0

- Replaced the `ncnn` engine (`realesrgan-ncnn-py`) with a new `onnx` engine (`onnxruntime`) -- the `ncnn` package turned out to be archived/unmaintained (as was every alternative checked in that same pybind11-around-ncnn-vulkan category), which also explained why it had no wheels for Python 3.13/3.14
- The `onnx` engine's model is converted once from the official `realesr-animevideov3.pth` weights (xinntao/Real-ESRGAN) and bundled directly in the package (~2.5 MB) -- no download, no external hosting, no sha256-pinning needed for it
- `onnxruntime` is now a base dependency (lightweight, actively maintained, wheels for every current Python version) rather than an install extra, so upscaling works out of the box with no extra install step; `opencv` remains the one opt-in extra
- `config.json`'s `upscaler` section: `{"engine": "ncnn", "ncnn_model": ...}` -> `{"engine": "onnx", "onnx_model": ...}` (default unchanged: the fast anime-tuned model)
- New `tools/convert_onnx_model.py` (behind a `convert` extra: `torch`, `onnx`) reproduces the source PyTorch architectures and regenerates the bundled ONNX models from their official weights, with a `comiccrawler`-style menu to pick which one; not needed to use the package
- Added `realesrgan-x4plus-anime-6b` (RRDBNet, heavier/higher-quality) as a second bundled `onnx_model` choice alongside the default `realesr-animevideov3-x4`
- `__version__` is now read from installed package metadata (`importlib.metadata`) instead of being hand-duplicated in `__init__.py` -- `pyproject.toml` is the only place the version is written

## 0.4.0

- Pluggable super-resolution backends -- `upscaler.py` now talks to a small `SREngine` interface (`sr_engine.py`) instead of hard-coding OpenCV, so adding another engine doesn't touch the upscaler itself
- New `ncnn` engine (`realesrgan-ncnn-py`), tuned for illustration/anime-style art and much faster than EDSR on CPU -- now the default (`realesr-animevideov3-x4`)
- `config.json`'s `model_name`/`model_scale` became a nested, engine-discriminated `upscaler` section (`{"engine": "opencv"|"ncnn", ...}`)
- OpenCV and ncnn are now separate optional install extras (`pip install .[opencv]` / `.[ncnn]`) instead of a hard dependency, so a download-only install no longer needs either
- Resume/parsing bug fixes in the downloader (page-number parsing, lexicographic sort overflow) and upscaler (gap-tolerant resume, per-page failure isolation), config path/reserved-name validation, worker-count sanity cap, dedup cleanup
- A failing engine `prepare()` step now fails fast in the main process with one clear error, instead of only surfacing inside a `ProcessPoolExecutor` initializer where it broke the whole worker pool and buried the real error under a `BrokenProcessPool` per page

## 0.3.0

- Package refresh -- pydantic config
- Installable via `pyproject.toml`
- Automatic model download (with sha256 integrity checks)
- Menu-driven launcher
- Parallel upscaling
- Test suite
- Bug fixes

## 0.2.0

- Bug fixes

## 0.1.0-alpha

- First release
