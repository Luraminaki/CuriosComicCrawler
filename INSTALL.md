# Install

## Prerequisites

- **Python 3.11 or 3.12** if you intend to use the `ncnn` engine (recommended, see below):
  `realesrgan-ncnn-py` only publishes prebuilt wheels for Python 3.8-3.12, with no source
  distribution to build from, so 3.13/3.14 will fail with `No matching distribution found
  for realesrgan-ncnn-py`. The `opencv` engine alone has no such ceiling; only pick a newer
  Python if you're sure you'll stick to that engine. Get Python from
  [python.org](https://www.python.org/downloads/) (Windows/macOS) or your distro's package
  manager (Linux); on Windows, `py -0` lists every version already installed, and
  `py -3.12 -m venv .venv` targets one specifically if `python`/`py` defaults to something
  newer.
- **A C library OpenCV needs at runtime**: whichever OpenCV wheel you end up with (see
  "Choose an upscaling engine" below) ships pre-built, but on Linux it still dynamically
  loads `libGL.so.1`, which isn't always installed on minimal/headless systems. See the
  Linux sections below if you hit an `ImportError: libGL.so.1: cannot open shared object
  file` when running the tools.

You do **not** need to manually download the super-resolution models anymore -- the
`opencv` engine downloads whichever model/scale your `config.json` asks for the first time
it's needed, and reuses it afterwards. The `ncnn` engine's models are bundled inside its
package instead, so there's nothing to download for that one either.

## Choose an upscaling engine

`comiccrawler-download` (page downloading only) needs nothing beyond the base install. But
upscaling (`comiccrawler-upscale`, or `comiccrawler` in `upscale`/`both` mode) needs one of
two optional extras, matching the `engine` you pick in `config.json`'s `upscaler` section:

| Extra    | `upscaler.engine` | What it installs                                                        |
|----------|--------------------|--------------------------------------------------------------------------|
| `ncnn`   | `"ncnn"`           | `realesrgan-ncnn-py` -- illustration/anime-tuned models, CPU-only, no Vulkan driver needed. **Recommended** for comic pages, and much faster than `opencv`/EDSR. |
| `opencv` | `"opencv"`         | `opencv-contrib-python` -- classic photo-trained models (EDSR/ESPCN/FSRCNN/LapSRN).           |

Install whichever one matches the engine you intend to use, e.g. `pip install -e ".[ncnn]"`.
The shipped `config.json` defaults to `"ncnn"`.

> [!WARNING]
> Installing **both** extras in the same environment can break the `opencv` engine:
> `realesrgan-ncnn-py` depends on plain `opencv-python`, which conflicts with
> `opencv-contrib-python` -- both ship files under the same `cv2/` package, and whichever
> installs last silently wins, which can delete `cv2.dnn_superres`. If you do want both
> engines available, reinstall contrib last:
> ```sh
> pip install --force-reinstall --no-deps opencv-contrib-python
> ```
> This is a known upstream packaging issue (`opencv-python` and `opencv-contrib-python`
> were never meant to coexist in one environment), not something specific to this project.

## Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -U -e ".[ncnn]"
```

`py -3.12` picks Python 3.12 explicitly, regardless of what `python`/`py` alone would default
to (see the `ncnn` note above) -- swap in `-3.11` if that's what you have installed instead.

## Linux -- Debian / Ubuntu

```sh
sudo apt update
sudo apt install -y python3-venv libgl1

python3 -m venv .venv
source .venv/bin/activate
pip install -U -e ".[ncnn]"
```

## Linux -- Arch

```sh
sudo pacman -S --needed python libglvnd

python -m venv .venv
source .venv/bin/activate
pip install -U -e ".[ncnn]"
```

## Optional: development tools

Linting (`ruff`), type checking (`basedpyright`), and the test suite (`pytest`) are
provided as an optional dependency group -- paired here with the `opencv` extra since the
test suite exercises that engine's real code path (the `ncnn` engine's tests fake out its
dependency instead, so installing it isn't required for `pytest` to pass):

```sh
pip install -U -e ".[dev,opencv]"
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
