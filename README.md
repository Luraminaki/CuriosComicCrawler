# CC DL-UPSCALER

I love [Sequential Art](https://www.collectedcurios.com/). But I don't always have a stable connection (And it seems the author has some issues with his server every once in a while)... And my sight isn't as good as it used to be... So I made this... Please bear in mind that I made this for entertaining and educative purpose only. Consider [supporting](https://www.patreon.com/collectedcurios) the author of this comic if you like his work.
Just edit the `config.json` file if you want to play around with the scripts...

## Version

The current version lives in [pyproject.toml](pyproject.toml). See [CHANGELOG.md](CHANGELOG.md) for the full version history.

## Table of content

<!-- TOC -->

- [CC DL-UPSCALER](#cc-dl-upscaler)
  - [Version](#version)
  - [Table of content](#table-of-content)
  - [Install](#install)
  - [Configuration](#configuration)
    - [About the models](#about-the-models)
    - [Posterisation](#posterisation)
    - [Workers and memory](#workers-and-memory)
  - [Usage](#usage)
    - [Launcher (menu-driven)](#launcher-menu-driven)
    - [Download comic pages](#download-comic-pages)
    - [Upscale downloaded pages](#upscale-downloaded-pages)
    - [Stopping a run (Ctrl+C)](#stopping-a-run-ctrlc)
  - [Outputs and logs](#outputs-and-logs)
  - [Exit codes](#exit-codes)
  - [Development](#development)

<!-- /TOC -->

## Install

See [INSTALL.md](INSTALL.md) for step-by-step instructions (Windows, Debian/Ubuntu, Arch).

## Configuration

Every command reads `config.json` (or a file passed via `-c/--configuration`), validated
against a pydantic model -- an invalid or missing field is reported as a single error message
(no traceback) and the command exits with code 1, instead of failing deep inside the code.

| Field                 | Type | Default | Meaning and constraints |
|-----------------------|------|---------|--------------------------|
| `root_site`           | string | required | Base URL comic pages are downloaded from (page URL = `root_site` + filename). Must be an `http`/`https` URL with a host, ending in `/` |
| `BD_name`             | string | required | Comic identifier, used as filename prefix and subfolder name. Must be a single path segment: not empty, not `.`/`..`, no `/` or `\`, not a Windows device name (`CON`, `PRN`, `AUX`, `NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`) |
| `padded`              | integer | required | Zero-padding width of the page number in filenames. Greater than 0 |
| `fails`               | integer | required | Number of *consecutive* missing filename variants that ends a run (a fully-missing page costs one per entry in `filename_variants`; a saved page resets the count). Greater than 0 |
| `filename_variants`   | list of strings | required | Ordered filename suffixes tried for each page, e.g. `["small", "small_b"]` tries `..._small.jpg` then `..._small_b.jpg`. At least one entry; entries may not be empty or contain `/` or `\` |
| `ext`                 | string | required | Image extension. Must start with `.` (e.g. `".jpg"`) |
| `folder_data`         | string | required | Root data folder, relative to the current directory |
| `folder_save_dl`      | string | required | Subfolder of `folder_data` downloaded pages are saved to |
| `folder_save_upscale` | string | required | Subfolder of `folder_data` upscaled pages are saved to |
| `folder_models`       | string | required | Subfolder of `folder_data` super-resolution models are stored in (used by the `opencv` engine only) |
| `upscaler`            | object | required | Which super-resolution engine/model to upscale with: `engine` is `"onnx"` or `"opencv"` -- see [About the models](#about-the-models) |
| `gray_values`         | integer | required | Number of gray levels the upscaled image is posterised to. 2 to 255 (255 leaves the image unchanged) |
| `posterise_method`    | string | `"histogram"` | How those gray levels are found: `"histogram"` or `"kmeans"` -- see [Posterisation](#posterisation) |
| `headers`             | object (string to string) | required | HTTP headers sent with every download request. Only advertise encodings `requests` can decode: the shipped `"Accept-Encoding": "gzip, deflate"` |
| `upscale_workers`     | integer or `null` | `null` (1 worker) | Number of pages upscaled in parallel, one worker process each. At least 1; capped at the available CPUs (and at the number of pages left) -- see [Workers and memory](#workers-and-memory) |

All four `folder_*` fields must be non-empty and may not contain a `..` segment.

### About the models

Upscaling is pluggable: `upscaler.engine` picks which super-resolution backend processes each
page. `upscaler.py` only talks to a small `SREngine` interface (`sr_engine.py`); each engine
lives in its own `sr_engine_<name>.py` module, so adding another one doesn't touch the upscaler
itself.

**`"engine": "onnx"`** (default, recommended) -- [ONNX Runtime](https://onnxruntime.ai/) models,
mostly tuned for illustration/anime-style art (a better fit for comic pages than EDSR, and much
faster). Needs no install extra: `onnxruntime` is a base dependency, and each model (converted
once from official [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) weights via
[tools/convert_onnx_model.py](tools/convert_onnx_model.py)) ships inside this package under
`src/curios_comic_crawler/assets/` -- nothing to download, no GPU/Vulkan driver needed. This
replaced an earlier `ncnn`-based engine that turned out to depend on an archived, unmaintained
package; see [CHANGELOG.md](CHANGELOG.md).

```json
"upscaler": {"engine": "onnx", "onnx_model": "realesr-animevideov3-x4"}
```

| `onnx_model`                  | Notes                                                       |
|--------------------------------|--------------------------------------------------------------|
| `realesr-animevideov3-x4`     | Default if omitted. SRVGGNetCompact, ~2.5 MB, the fast option |
| `realesrgan-x4plus-anime-6b`  | RRDBNet (6 blocks), ~17 MB, heavier/slower but potentially higher quality |
| `realesr-general-x4v3`        | SRVGGNetCompact, ~4.9 MB, general-purpose, same speed class as the default |
| `realesrgan-x4plus`           | RRDBNet (23 blocks), ~64 MB, general-purpose (photo, not anime-tuned), the heaviest/slowest option |

All four are bundled -- there's no separate download step for any of them. More models can be
converted with `tools/convert_onnx_model.py` (see [tools/README.md](tools/README.md)); wiring a
newly converted one in as an `onnx_model` choice means also adding it to `models.py`'s
`OnnxModelName` and `sr_engine_onnx.py`'s `_MODEL_FILENAMES`.

**`"engine": "opencv"`** -- OpenCV's `dnn_superres` module, trained on general photos. No extra install needed (`opencv-contrib-python` is a base dependency, needed regardless of engine since posterising/sharpening always run through `cv2`).

```json
"upscaler": {"engine": "opencv", "model_name": "edsr", "model_scale": 3}
```

| Field         | Meaning                                                                 |
|---------------|--------------------------------------------------------------------------|
| `model_name`  | Model family: `edsr`, `espcn`, `fsrcnn`, `fsrcnn-small`, or `lapsrn`      |
| `model_scale` | Upscaling factor for `model_name` (2/3/4, or 2/4/8 for `lapsrn`); any other pair is rejected, with the known scales in the message |

Model weights come from [EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow), [TF-ESPCN](https://github.com/fannymonori/TF-ESPCN), [FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow), and [TF-LapSRN](https://github.com/fannymonori/TF-LapSRN). You don't need to download these by hand: the upscaler fetches whichever `model_name`/`model_scale` combination `config.json` asks for into `data/models/` the first time it's needed, and reuses it on later runs. Every model file's sha256 is pinned in `model_registry.py` and checked both right after downloading and before reusing an already-cached file, so a truncated download or a `.pb` corrupted on disk gets re-fetched automatically instead of silently being fed to OpenCV. A download goes to a uniquely named temporary file that only replaces the model once its sha256 matches, so two runs fetching the same model at once can't corrupt it.

OpenCV is pinned below 5 (`opencv-contrib-python>=4.9,<5`): OpenCV 5 can no longer read these TensorFlow `.pb` models.

### Posterisation

After upscaling, every page is reduced to `gray_values` gray levels, then sharpened.
`posterise_method` picks how the levels are found; both start from the same levels and their
results differ by at most 9/255, in the light grays (no visible difference side by side).
Measured on a real page (414x1284 upscaled to 1656x5136):

| `posterise_method`      | How                                                  | Time per page | Extra memory |
|-------------------------|------------------------------------------------------|---------------|--------------|
| `"histogram"` (default) | k-means on the 256-value histogram, then a lookup table | ~0.003 s   | ~0 MiB       |
| `"kmeans"`              | OpenCV's k-means on every pixel (the method before 0.7.0) | ~2.6 s   | +189 MiB     |

### Workers and memory

Pages are upscaled in worker processes (`upscale_workers`, default 1). Each worker loads its
own copy of the model, and a single worker already uses every physical core through ONNX
Runtime / OpenCV threads (the threads are split across physical cores, not hyperthreads:
same wall time, half the CPU). More workers cost a lot of memory for little speed. Measured
on 24 real pages with the default model:

| Workers | Wall time (`histogram`) | Wall time (`kmeans`) |
|---------|-------------------------|----------------------|
| 1       | 31.5 s                  | 101.8 s              |
| 2       | 27.5 s                  | 90.7 s               |
| 4       | 26.3 s                  | 88.4 s               |

Peak memory per worker on a full page: ~382 MiB with the light models
(`realesr-animevideov3-x4`, `realesr-general-x4v3`), ~4.2 GiB (4218 MiB) with the RRDBNet ones
(`realesrgan-x4plus-anime-6b`, `realesrgan-x4plus`). A worker dying (usually out of memory with
a heavy model or several workers) is reported once, and the remaining pages count as failed.

Workers are started with `forkserver` (`spawn` on Windows), the same on every Python version;
the forkserver adds one resident helper process (~125 MiB RSS, mostly shared libraries).

## Usage

### Launcher (menu-driven)

```sh
comiccrawler --configuration=config.json
```

Asks what to do (download, upscale, or both) and, for whichever step you pick, whether to
force-reprocess everything -- re-download every page from scratch, and/or re-upscale every
downloaded page, ignoring what's already there.

It also runs non-interactively, e.g. for a cron job that redoes everything from scratch:

```sh
comiccrawler -c config.json --mode both --force-download --force-upscale
```

`--mode` (`-m`) accepts `download`, `upscale`, or `both`; passing it skips the interactive menu
entirely (and the `--force-*` flags default to off unless given). A `--force-*` flag given
*without* `--mode` answers its own question in the menu, so only the other questions are asked.
With no `--mode` and no terminal to ask from (cron, a pipe), the launcher exits with code 2
instead of waiting for input. A download-only run doesn't load OpenCV or ONNX Runtime
(~94 MiB less memory).

### Download comic pages

```sh
comiccrawler-download --configuration=config.json
```

Resumes after the highest page number already saved, and fetches pages in order, trying each
entry of `filename_variants` until one is saved:

- HTTP 404 or 410 means the variant doesn't exist; `fails` consecutive missing variants mean
  the end of the comic, and the run ends successfully (exit code 0).
- HTTP 429, 5xx and network errors (including a connection dropping mid-download) are retried
  up to 3 attempts, waiting the server's `Retry-After` (capped at 60 s) or else 2 s, then 4 s.
- Any other status, an answer that isn't an image, or errors that outlast the retries stop the
  run with exit code 1.

The whole run uses one HTTP session (one connection, `headers` sent with every request), waits
1 s between requests, and writes each page to a `.part` file that is renamed only once complete.

Pass `--force` to ignore what's already downloaded and start over from page 1 (existing files are
overwritten).

### Upscale downloaded pages

```sh
comiccrawler-upscale --configuration=config.json
```

Processes every downloaded page that has no upscaled file of the same name yet. Each page is
upscaled, posterised, sharpened, and written atomically (never left truncated). A page that
fails is logged and the run carries on with the others, then exits with code 1; an unexpected
error on a page (e.g. `MemoryError`) stops the run at once, cancelling the pages not started
yet.

Pass `--force` to re-upscale every downloaded page, ignoring what's already upscaled.

All three also work as `python -m curios_comic_crawler.cli_launcher` / `cli_download` /
`cli_upscale` if you'd rather not rely on the installed console scripts, all default to
`./config.json` when `--configuration` (`-c`) is omitted, and all accept `--version` and
`--help`.

### Stopping a run (Ctrl+C)

Ctrl+C stops any command cleanly with exit code 130 and an `Interrupted` log line, without a
traceback. While upscaling, the pages already running finish and are written, the queued ones are
cancelled (the log says how many of each), and no truncated file is left behind; the next run
picks up where this one stopped.

## Outputs and logs

With the shipped `config.json` (paths are relative to the current directory):

| What | Where |
|------|-------|
| Downloaded pages | `<folder_data>/<folder_save_dl>/<BD_name>/<BD_name>_<page>_<variant><ext>`, e.g. `data/DL/SA/SA_0042_small.jpg` (`<page>` zero-padded to `padded` digits) |
| Upscaled pages   | Same filename under `<folder_data>/<folder_save_upscale>/<BD_name>/`, e.g. `data/UPSCALE/SA/SA_0042_small.jpg` |
| OpenCV models    | `<folder_data>/<folder_models>/`, e.g. `data/models/EDSR_x3.pb` (`opencv` engine only) |
| Logs             | `<command>-<pid>.log` in the current directory, e.g. `comiccrawler-download-4184.log` |

Every command logs both to the console and to its own log file (one per run; worker processes
log into it too). A log file rotates at 5 MiB, keeping 5 older parts (`.log.1` ... `.log.5`), and
only the 10 most recent runs of each command are kept -- older ones are deleted at startup.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | Success: nothing left to do, or the download reached the end of the comic |
| 1    | Failure: the download stopped on a network/server error, a page failed to upscale, a worker died, the configuration is missing or invalid, or the command crashed (traceback in the log) |
| 2    | `comiccrawler` run without `--mode` and without a terminal to ask from (an invalid command-line argument also exits 2, from `argparse`) |
| 130  | Interrupted (Ctrl+C) |

With `--mode both`, the upscale step runs even if the download step failed, and the launcher
returns the higher of the two steps' codes.

## Development

```sh
pip install -U -e ".[dev]"
ruff check .
ruff format --check .
basedpyright                                    # src/, tests/ and tools/, per pyproject.toml
pylint --recursive=y src tests tools
radon cc -nc src tests tools                    # prints nothing when every function is grade A/B
vulture --min-confidence 80 --ignore-names cls src tests tools
pytest
```

- `basedpyright` type-checks `tools/` too, which imports `torch` and `onnx`: install the `convert`
  extra as well (`pip install -U -e ".[dev,convert]"`) for a clean run. Nothing else needs it.
- `vulture` also lists imports used only in quoted type annotations (under `if TYPE_CHECKING:`);
  those are false positives -- unused imports are `ruff`'s job.
- `pytest` runs 321 tests in about 1.6 s, with no network access: an autouse guard makes any
  network connection fail the test. Besides behaviour, convention tests enforce Google-style
  docstrings, no unused code, no duplicated code and a radon complexity of grade B or better.

See [tools/README.md](tools/README.md) for the script that regenerates the bundled ONNX models
from their official source weights -- not needed to use or develop the package, only if you're
changing which models are bundled.
