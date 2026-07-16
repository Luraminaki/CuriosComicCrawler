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
  - [Usage](#usage)
    - [Launcher (menu-driven)](#launcher-menu-driven)
    - [Download comic pages](#download-comic-pages)
    - [Upscale downloaded pages](#upscale-downloaded-pages)
  - [Development](#development)

<!-- /TOC -->

## Install

See [INSTALL.md](INSTALL.md) for step-by-step instructions (Windows, Debian/Ubuntu, Arch).

## Configuration

Both tools read `config.json` (or a file passed via `-c/--configuration`), validated
against a pydantic model -- an invalid or missing field is reported clearly instead of
failing deep inside the code.

| Field                 | Meaning                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `root_site`           | Base URL comic pages are downloaded from                                 |
| `BD_name`             | Comic identifier, used as filename prefix and subfolder name             |
| `padded`              | Zero-padding width of the page number in filenames                       |
| `fails`               | Number of *consecutive* failed download attempts that ends a run (a fully-missing page costs one attempt per entry in `filename_variants`) |
| `filename_variants`   | Ordered list of filename suffixes tried for each page, e.g. `["small", "small_b"]` tries `..._small.jpg` then `..._small_b.jpg` |
| `ext`                 | Image extension, including the leading dot                               |
| `folder_data`         | Root data folder                                                         |
| `folder_save_dl`      | Subfolder downloaded pages are saved to                                  |
| `folder_save_upscale` | Subfolder upscaled pages are saved to                                    |
| `folder_models`       | Subfolder super-resolution models are stored in (used by the `opencv` engine only) |
| `upscaler`            | Which super-resolution engine/model to upscale with -- see below         |
| `gray_values`         | Number of posterisation clusters applied to the upscaled image           |
| `headers`             | HTTP headers sent with every download request                           |
| `upscale_workers`     | *(optional)* Number of pages to upscale in parallel (one worker process per page in flight). Omit or set to `null` to use one worker per CPU core. |

### About the models

Upscaling is pluggable: `upscaler.engine` picks which super-resolution backend processes each
page. `upscaler.py` only talks to a small `SREngine` interface (`sr_engine.py`); each engine
lives in its own `sr_engine_<name>.py` module, so adding another one doesn't touch the upscaler
itself.

**`"engine": "ncnn"`** (default, recommended) -- [realesrgan-ncnn-py](https://github.com/Tohrusky/realesrgan-ncnn-py), tuned for illustration/anime-style art (a better fit for comic pages than the general-photo OpenCV models below, and much faster than EDSR -- see the comparison in [CHANGELOG.md](CHANGELOG.md)). Requires the `ncnn` extra: `pip install -e ".[ncnn]"`. Runs on CPU only -- no Vulkan driver needed. Models are bundled inside the package, so there's nothing to download. See [INSTALL.md](INSTALL.md) for a packaging conflict to watch for if you also install the `opencv` extra.

```json
"upscaler": {"engine": "ncnn", "ncnn_model": "realesr-animevideov3-x4"}
```

| `ncnn_model`               | Notes                                  |
|----------------------------|-----------------------------------------|
| `realesr-animevideov3-x2`  | Anime-tuned, 2x                         |
| `realesr-animevideov3-x3`  | Anime-tuned, 3x                         |
| `realesr-animevideov3-x4`  | Anime-tuned, 4x -- the sweet spot: as good as the others below, ~10x faster |
| `realesrgan-x4plus-anime`  | Anime-tuned, 4x                         |
| `realesrgan-x4plus`        | General-purpose (photo) model, 4x       |

**`"engine": "opencv"`** -- OpenCV's `dnn_superres` module, trained on general photos. Requires the `opencv` extra: `pip install -e ".[opencv]"`.

```json
"upscaler": {"engine": "opencv", "model_name": "edsr", "model_scale": 3}
```

| Field         | Meaning                                                                 |
|---------------|--------------------------------------------------------------------------|
| `model_name`  | Model family: `edsr`, `espcn`, `fsrcnn`, `fsrcnn-small`, or `lapsrn`      |
| `model_scale` | Upscaling factor for `model_name` (2/3/4, or 2/4/8 for `lapsrn`)          |

Model weights come from [EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow), [TF-ESPCN](https://github.com/fannymonori/TF-ESPCN), [FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow), and [TF-LapSRN](https://github.com/fannymonori/TF-LapSRN). You don't need to download these by hand: the upscaler fetches whichever `model_name`/`model_scale` combination `config.json` asks for into `data/models/` the first time it's needed, and reuses it on later runs. Every model file's sha256 is pinned in `model_registry.py` and checked both right after downloading and before reusing an already-cached file, so a truncated download or a `.pb` corrupted on disk gets re-fetched automatically instead of silently being fed to OpenCV.

Upscaling itself runs across multiple worker processes (see `upscale_workers` above) -- each page is independent, so this uses all CPU cores by default instead of processing pages one at a time.

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

`--mode` accepts `download`, `upscale`, or `both`; passing it skips the interactive menu
entirely (and the `--force-*` flags default to off unless given).

### Download comic pages

```sh
comiccrawler-download --configuration=config.json
```

Pass `--force` to ignore what's already downloaded and start over from page 1.

### Upscale downloaded pages

```sh
comiccrawler-upscale --configuration=config.json
```

Pass `--force` to re-upscale every downloaded page, ignoring what's already upscaled.

All three also work as `python -m curios_comic_crawler.cli_launcher` / `cli_download` / `cli_upscale` if you'd rather not rely on the installed console scripts, and all default to `./config.json` when `--configuration` is omitted.

## Development

```sh
pip install -U -e ".[dev,opencv]"
ruff check src/ tests/
basedpyright src/
pytest
```

(paired with the `opencv` extra since the test suite exercises that engine's real code path;
the `ncnn` engine's tests fake out its dependency instead, so installing it isn't required.)
