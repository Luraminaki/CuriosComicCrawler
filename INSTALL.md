# Install

## Prerequisites

- **Python 3.11+**. Get it from [python.org](https://www.python.org/downloads/) (Windows/macOS)
  or your distro's package manager (Linux).
- **A C library OpenCV needs at runtime**: the `opencv-contrib-python` wheel ships pre-built,
  but on Linux it still dynamically loads `libGL.so.1`, which isn't always installed on
  minimal/headless systems. See the Linux sections below if you hit an `ImportError:
  libGL.so.1: cannot open shared object file` when running the tools. `cv2` is needed
  regardless of which upscale engine you use -- posterising/sharpening always run through it.

You do **not** need to manually download the super-resolution models -- the `opencv` engine
downloads whichever model/scale your `config.json` asks for the first time it's needed, and
reuses it afterwards. The `onnx` engine's models are bundled inside this package instead, so
there's nothing to download for those either.

## Choose an upscaling engine

Both upscale engines (`upscaler.engine` in `config.json`) work out of the box with a plain base
install -- no extras needed for either one:

| Engine (`upscaler.engine`) | What it uses                                                        |
|-----------------------------|------------------------------------------------------------------------|
| `"onnx"` (default)          | Small bundled [ONNX Runtime](https://onnxruntime.ai/) models tuned for illustration/anime-style art. CPU-only, no GPU/Vulkan driver needed. |
| `"opencv"`                  | `opencv-contrib-python` -- classic photo-trained models (EDSR/ESPCN/FSRCNN/LapSRN), downloaded on first use. |

The shipped `config.json` defaults to `"onnx"`.

`comiccrawler-download` (page downloading only) doesn't need either engine's dependencies, but
they're installed regardless since this project doesn't split its base install that finely.

## Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U -e .
```

## Linux -- Debian / Ubuntu

```sh
sudo apt update
sudo apt install -y python3-venv libgl1

python3 -m venv .venv
source .venv/bin/activate
pip install -U -e .
```

## Linux -- Arch

```sh
sudo pacman -S --needed python libglvnd

python -m venv .venv
source .venv/bin/activate
pip install -U -e .
```

## Optional: development tools

Linting (`ruff`), type checking (`basedpyright`), and the test suite (`pytest`) are
provided as an optional dependency group:

```sh
pip install -U -e ".[dev]"
ruff check src/ tests/ tools/
basedpyright src/
pytest
```

## Optional: regenerating the bundled ONNX models

Only needed if you're changing which models `sr_engine_onnx.py` bundles -- see
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
