# Changelog

All notable changes to this project are documented here. Versioning loosely
follows [Semantic Versioning](https://semver.org/).

## 0.7.0 - 2026-11-26

### Added

- New `posterise_method` setting: `"histogram"` (default) runs k-means on the 256-value histogram (~0.003 s per page, no extra memory); `"kmeans"` keeps the previous per-pixel OpenCV k-means (~2.6 s per page, +189 MiB). Results differ by a few levels in the light grays (8-13/255 at most on most pages); the histogram method always gives every level asked for; gray levels are now rounded instead of truncated
- `--version` on all three commands
- Documented exit codes: 0 success or end of the comic, 1 failure (download stopped on a network/server error, a page failed to upscale, invalid configuration, crash), 2 launcher run without `--mode` and without a terminal, 130 interrupted
- The launcher's `--force-download`/`--force-upscale` given without `--mode` now answer their own question in the menu; only the other questions are asked
- Stricter configuration checks: `root_site` must be an `http`/`https` URL with a host, ending in `/`; `filename_variants` entries may not be empty or contain `/` or `\`; `ext` must start with `.`; `folder_*` fields may not be empty or contain `..`

### Changed

- `upscale_workers` now defaults to 1 worker instead of one per CPU: each worker loads its own model, and 2-4 workers were only slightly faster (24 pages: 31.5 s with 1, 27.5 s with 2, 26.3 s with 4) for several times the memory
- Lower upscaling memory: ONNX Runtime's memory arena is off and the model input/output conversions make fewer copies -- peak per worker 768 -> 382 MiB with the light models, 7080 -> 4218 MiB with the RRDBNet ones, same speed
- Faster model output conversion (6x, 24 MiB less peak): a 50-page run went from 32.6 to 29.5 s, worker peak from 347 to 307 MiB
- Worker threads are split across physical cores instead of hyperthreads (same wall time, half the CPU time); the CPU count honours CPU affinity
- Upscale workers start with `forkserver` (`spawn` on Windows) on every Python version, instead of forking a process that runs the log thread; worker logs now reach the log file
- Downloader: one HTTP session for the whole run; HTTP 429, 5xx and network errors (including a dropped connection mid-download) are retried, honouring `Retry-After` (capped at 60 s) or waiting 2 s then 4 s; only 404/410 mean a missing page; any other status or a non-image answer stops the run with exit code 1
- The shipped `config.json` only advertises `"Accept-Encoding": "gzip, deflate"` (Brotli/zstd answers can't be decoded)
- A download-only launcher run no longer loads OpenCV and ONNX Runtime (~94 MiB less)
- A configuration error is reported as one message without a traceback; unknown keys (e.g. a misspelt `posterize_method`) are now errors instead of being ignored
- A model that can't be downloaded or verified is one error line and exit code 1, not a crash
- Python 3.11 is the minimum supported version; the `dev` extra adds `pylint`, `radon` and `vulture`; the `convert` extra needs `torch>=2.5`
- Test suite: 321 tests in ~1.6 s, with convention tests (docstrings, unused code, duplication, complexity) and a guard that fails any test opening a network connection
- Old log runs are pruned by modification time instead of process ID
- `logreset` removed (no longer needed)

### Fixed

- The `opencv` engine failed on fresh installs: OpenCV 5 can't read its `.pb` models, so OpenCV is now pinned to `>=4.9,<5`
- A single HTTP error (e.g. a 503 from a flaky server) silently ended the download as if the comic were complete, with exit code 0
- An HTML error page could be saved as a `.jpg`, breaking every later upscale; non-image answers are now refused, and pages are written to a `.part` file renamed only when complete
- Ctrl+C: every command now exits with code 130 and no traceback (Ctrl+D at the launcher's questions too); workers ignore Ctrl+C from the moment they start; the upscaler lets running pages finish, cancels the queued ones and never leaves a truncated page (which the next run would have skipped as done)
- An unexpected error on one page now stops the upscale run instead of processing every remaining page first
- A failed page write was counted as done; a dead worker is now reported once, with its cause, instead of once per remaining page
- `filename_variants` such as `"../x"`, and absolute or `..\x` sub-folders (`folder_save_dl`, `folder_save_upscale`, `folder_models`), wrote outside the data folder
- After a worker died, a page written just before could be counted as failed; the interrupt summary now also counts failed pages
- Temporary files of pages, models and the converter share one scheme, unique per process, so two runs on one folder can't mix their writes
- Model downloads: two runs fetching the same model could corrupt it, and a failed download left a `.part` file behind; downloads now go to a unique temporary file, checked by SHA-256, then renamed
- `tools/convert_onnx_model.py` loads weights with `weights_only=True`, streams the download instead of holding it in memory, and only replaces a bundled model once the export passed its check
- The downloader no longer sleeps twice on a fully-missing page

## 0.6.0

- Added the two remaining convertible models as real `upscaler.onnx_model` choices, for exhaustivity: `realesr-general-x4v3` (SRVGGNetCompact, ~4.9 MB, general-purpose) and `realesrgan-x4plus` (RRDBNet 23-block, ~64 MB, general-purpose photo, the heaviest/slowest option) -- all four models `tools/convert_onnx_model.py` can produce are now bundled and wired in

## 0.5.0

- Replaced the `ncnn` engine (`realesrgan-ncnn-py`) with a new `onnx` engine (`onnxruntime`) -- the `ncnn` package turned out to be archived/unmaintained (as was every alternative checked in that same pybind11-around-ncnn-vulkan category), which also explained why it had no wheels for Python 3.13/3.14
- The `onnx` engine's model is converted once from the official `realesr-animevideov3.pth` weights (xinntao/Real-ESRGAN) and bundled directly in the package (~2.5 MB) -- no download, no external hosting, no sha256-pinning needed for it
- `onnxruntime` is now a base dependency (lightweight, actively maintained, wheels for every current Python version) rather than an install extra, so upscaling works out of the box with no extra install step; `opencv-contrib-python` also stays a base dependency (not an extra) since posterising/sharpening always run through `cv2` regardless of which engine produced the upscaled image -- a brief attempt to gate it behind an `opencv` extra broke the default `onnx` engine on a plain install
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
