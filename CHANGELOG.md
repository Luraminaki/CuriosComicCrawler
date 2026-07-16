# Changelog

All notable changes to this project are documented here. Versioning loosely
follows [Semantic Versioning](https://semver.org/).

## 0.4.0

- Pluggable super-resolution backends -- `upscaler.py` now talks to a small `SREngine` interface (`sr_engine.py`) instead of hard-coding OpenCV, so adding another engine doesn't touch the upscaler itself
- New `ncnn` engine (`realesrgan-ncnn-py`), tuned for illustration/anime-style art and much faster than EDSR on CPU -- now the default (`realesr-animevideov3-x4`)
- `config.json`'s `model_name`/`model_scale` became a nested, engine-discriminated `upscaler` section (`{"engine": "opencv"|"ncnn", ...}`)
- OpenCV and ncnn are now separate optional install extras (`pip install .[opencv]` / `.[ncnn]`) instead of a hard dependency, so a download-only install no longer needs either
- Resume/parsing bug fixes in the downloader (page-number parsing, lexicographic sort overflow) and upscaler (gap-tolerant resume, per-page failure isolation), config path/reserved-name validation, worker-count sanity cap, dedup cleanup

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
