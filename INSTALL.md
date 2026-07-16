# Install

## Prerequisites

- **Python 3.11+**. Get it from [python.org](https://www.python.org/downloads/) (Windows/macOS)
  or your distro's package manager (Linux).
- **A C library OpenCV needs at runtime, only if you use the `opencv` engine** (see "Choose an
  upscaling engine" below): the `opencv-contrib-python` wheel ships pre-built, but on Linux it
  still dynamically loads `libGL.so.1`, which isn't always installed on minimal/headless
  systems. See the Linux sections below if you hit an `ImportError: libGL.so.1: cannot open
  shared object file` when running the tools. The default `onnx` engine has no such dependency.

You do **not** need to manually download the super-resolution models -- the `opencv` engine
downloads whichever model/scale your `config.json` asks for the first time it's needed, and
reuses it afterwards. The `onnx` engine's model is bundled inside this package instead, so
there's nothing to download for that one either.

## Choose an upscaling engine

`comiccrawler-download` (page downloading only) needs nothing beyond the base install. Upscaling
(`comiccrawler-upscale`, or `comiccrawler` in `upscale`/`both` mode) works out of the box with
the default engine, or with one optional extra for the alternative:

| Engine (`upscaler.engine`) | Extra needed        | What it uses                                                        |
|-----------------------------|----------------------|------------------------------------------------------------------------|
| `"onnx"` (default)          | none -- base install | A small bundled [ONNX Runtime](https://onnxruntime.ai/) model tuned for illustration/anime-style art. CPU-only, no GPU/Vulkan driver needed. |
| `"opencv"`                  | `pip install -e ".[opencv]"` | `opencv-contrib-python` -- classic photo-trained models (EDSR/ESPCN/FSRCNN/LapSRN). |

The shipped `config.json` defaults to `"onnx"`, so a plain base install is enough to try it.

## Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U -e .
```

## Linux -- Debian / Ubuntu

```sh
sudo apt update
sudo apt install -y python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install -U -e .
```

(add `libgl1` to the `apt install` line above if you'll use the `opencv` extra -- see the
Prerequisites note.)

## Linux -- Arch

```sh
sudo pacman -S --needed python

python -m venv .venv
source .venv/bin/activate
pip install -U -e .
```

(add `libglvnd` to the `pacman` line above if you'll use the `opencv` extra -- see the
Prerequisites note.)

## Optional: development tools

Linting (`ruff`), type checking (`basedpyright`), and the test suite (`pytest`) are
provided as an optional dependency group -- paired here with the `opencv` extra so the test
suite can exercise that engine's real code path too (the `onnx` engine needs nothing extra,
since `onnxruntime` and its model are already part of the base install):

```sh
pip install -U -e ".[dev,opencv]"
ruff check src/ tests/ tools/
basedpyright src/
pytest
```

## Optional: regenerating the bundled ONNX model

Only needed if you're changing which model `sr_engine_onnx.py` bundles -- see
[tools/README.md](tools/README.md). Requires the `convert` extra (`pip install -e ".[convert]"`,
which pulls in `torch`, a large download) and is unrelated to installing or developing the
package otherwise.

## Verifying the install

```sh
comiccrawler-download --help
comiccrawler-upscale --help
```

Both should print their usage text. See [README.md](README.md) for how to configure
and run them.
