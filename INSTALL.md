# Install

## Prerequisites

- **Python 3.11+**. Get it from [python.org](https://www.python.org/downloads/) (Windows/macOS)
  or your distro's package manager (Linux).
- **A C library OpenCV needs at runtime**: the `opencv-contrib-python` wheel ships
  pre-built, but on Linux it still dynamically loads `libGL.so.1`, which isn't always
  installed on minimal/headless systems. See the Linux sections below if you hit an
  `ImportError: libGL.so.1: cannot open shared object file` when running the tools.

You do **not** need to manually download the super-resolution models anymore — the
upscaler downloads whichever model/scale your `config.json` asks for the first time
it's needed, and reuses it afterwards.

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
ruff check src/ tests/
basedpyright src/
pytest
```

## Verifying the install

```sh
comiccrawler-download --help
comiccrawler-upscale --help
```

Both should print their usage text. See [README.md](README.md) for how to configure
and run them.
