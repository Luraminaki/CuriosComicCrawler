#!/usr/bin/env python3
"""Typed configuration for CuriosComicCrawler.

Replaces raw `dict` access into a parsed `config.json` with a validated pydantic model.
"""

import json
import pathlib
import urllib.parse
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from curios_comic_crawler.model_registry import MODEL_MANIFEST
from curios_comic_crawler.models import ModelName, OnnxModelName, PosteriseMethod

if TYPE_CHECKING:
    from pydantic import ValidationInfo

_PATH_SEPARATORS = ("/", "\\")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


class OpenCVUpscaleConfig(BaseModel):
    """Upscale using OpenCV's `dnn_superres` module (see `sr_engine_opencv.py`).

    Attributes:
        model_name: Super-resolution model family to use.
        model_scale: Upscaling factor to use for `model_name`.
    """

    # A misspelt key (e.g. "posterize_method") would otherwise be dropped silently.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    engine: Literal["opencv"] = "opencv"
    model_name: ModelName
    model_scale: int

    @model_validator(mode="after")
    def _check_model_combo(self) -> "OpenCVUpscaleConfig":
        """Require a (model name, scale) pair that has a downloadable model.

        Returns:
            OpenCVUpscaleConfig: This config, unchanged.

        Raises:
            ValueError: If no model exists for the pair; the message lists the name's known scales.
        """
        if (self.model_name, self.model_scale) not in MODEL_MANIFEST:
            known_scales = sorted(
                scale for name, scale in MODEL_MANIFEST if name == self.model_name
            )
            raise ValueError(
                f"No known model for model_name={self.model_name!r}, "
                + f"model_scale={self.model_scale!r}. Known scales for {self.model_name!r}: "
                + f"{known_scales}"
            )
        return self


class OnnxUpscaleConfig(BaseModel):
    """Upscale using a bundled ONNX Runtime model (see `sr_engine_onnx.py`).

    No install extra needed -- `onnxruntime` is a base dependency, and the model ships inside
    this package. Runs on CPU only.

    Attributes:
        onnx_model: Which bundled model to use. Most are tuned for illustration/anime-style
            art; `realesrgan-x4plus` is the general-purpose (photo) exception.
    """

    # A misspelt key (e.g. "posterize_method") would otherwise be dropped silently.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    engine: Literal["onnx"] = "onnx"
    onnx_model: OnnxModelName = "realesr-animevideov3-x4"


UpscaleConfig = Annotated[OpenCVUpscaleConfig | OnnxUpscaleConfig, Field(discriminator="engine")]


class AppConfig(BaseModel):
    """Validated content of `config.json`.

    Attributes:
        root_site: Base URL comic pages are downloaded from. Kept as a plain `str` (not
            pydantic's `HttpUrl`) because the code builds page URLs with naive string
            concatenation (`root_site + image_name`); `HttpUrl` normalizes URLs in ways that
            would silently change that concatenation.
        bd_name: Comic identifier, also used as the image filename prefix and the
            download/upscale subfolder name. Must be a single path segment (no path
            separators, and not a Windows-reserved device name) since it is joined directly
            into paths.
        padded: Zero-padding width of the page number in image filenames.
        fails: Number of consecutive failed download attempts that ends a run. Each filename
            variant tried for a page counts as one attempt, so a fully-missing page costs as
            many fails as there are variants in `filename_variants`; any successful download
            resets the count to 0.
        filename_variants: Ordered list of filename suffixes tried for each page (e.g.
            `["small", "small_b"]` tries `..._small.jpg` then `..._small_b.jpg`). At least one
            entry is required.
        ext: Image file extension, including the leading dot (e.g. `".jpg"`).
        folder_data: Root data folder, relative to the current working directory.
        folder_save_dl: Subfolder (under `folder_data`) downloaded pages are saved to.
        folder_save_upscale: Subfolder (under `folder_data`) upscaled pages are saved to.
        folder_models: Subfolder (under `folder_data`) super-resolution models are stored in.
        upscaler: Which super-resolution engine/model to upscale with -- either
            `OpenCVUpscaleConfig` (`engine: "opencv"`) or `OnnxUpscaleConfig` (`engine: "onnx"`).
        gray_values: Number of posterisation clusters `area_posterise` reduces each image to.
        posterise_method: How the gray levels are found: `"histogram"` (k-means on the 256-value
            histogram, the default: fast and light) or `"kmeans"` (OpenCV's k-means on every
            pixel, ~2.6 s and ~190 MB more per page). Both give visually identical results.
        headers: HTTP headers sent with every download request.
        upscale_workers: Number of pages to upscale in parallel, each in its own process (every
            worker loads its own copy of the model, so this trades RAM for throughput). `None`
            (the default) uses one worker, which already uses every CPU; capped at the
            available CPUs.
    """

    root_site: str
    bd_name: str = Field(alias="BD_name")
    padded: int = Field(gt=0)
    fails: int = Field(gt=0)
    filename_variants: list[str] = Field(min_length=1)
    ext: str
    folder_data: str
    folder_save_dl: str
    folder_save_upscale: str
    folder_models: str
    upscaler: UpscaleConfig
    gray_values: int = Field(ge=2, le=255)
    posterise_method: PosteriseMethod = "histogram"
    headers: dict[str, str]
    upscale_workers: int | None = Field(default=None, ge=1)

    # A misspelt key (e.g. "posterize_method") would otherwise be dropped silently.
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("bd_name")
    @classmethod
    def _check_bd_name(cls, value: str) -> str:
        """Require a single, portable path segment (it names the saved files).

        Args:
            value (str): The `BD_name` value.

        Returns:
            str: `value`, unchanged.

        Raises:
            ValueError: If it's empty, `.`/`..`, contains a separator, or is reserved on Windows.
        """
        if not value or value in (".", "..") or any(sep in value for sep in _PATH_SEPARATORS):
            raise ValueError(f"BD_name must be a single path segment, not {value!r}")
        if value.upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"BD_name {value!r} is a reserved Windows device name")
        return value

    @field_validator("ext")
    @classmethod
    def _check_ext(cls, value: str) -> str:
        """Require the leading dot that filenames are built with.

        Args:
            value (str): The `ext` value.

        Returns:
            str: `value`, unchanged.

        Raises:
            ValueError: If it doesn't start with `.`.
        """
        if not value.startswith("."):
            raise ValueError(f"ext must start with a leading dot, not {value!r}")
        return value

    @field_validator("folder_data", "folder_save_dl", "folder_save_upscale", "folder_models")
    @classmethod
    def _check_data_folder(cls, value: str, info: "ValidationInfo") -> str:
        """Keep data folders non-empty and inside the folder they're joined to.

        Args:
            value (str): The folder setting.
            info (ValidationInfo): Which field is being validated (for the message).

        Returns:
            str: `value`, unchanged.

        Raises:
            ValueError: If it's empty or contains a `..` segment (either separator counts).
        """
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        # Windows paths split on both separators: a `..\\x` written for Windows is caught on
        # every platform.
        if ".." in pathlib.PureWindowsPath(value).parts:
            raise ValueError(f'{info.field_name} must not contain ".." path segments: {value!r}')
        return value

    @field_validator("folder_save_dl", "folder_save_upscale", "folder_models")
    @classmethod
    def _check_subfolder(cls, value: str, info: "ValidationInfo") -> str:
        """Keep the sub-folders relative: joined to an absolute path, `folder_data` is ignored.

        Args:
            value (str): The folder setting.
            info (ValidationInfo): Which field is being validated (for the message).

        Returns:
            str: `value`, unchanged.

        Raises:
            ValueError: If it's an absolute path, or has a drive letter (Windows).
        """
        if pathlib.PureWindowsPath(value).anchor:
            raise ValueError(f"{info.field_name} must be relative to folder_data: {value!r}")
        return value

    @field_validator("root_site")
    @classmethod
    def _check_root_site(cls, value: str) -> str:
        """Require an http(s) URL with a host, ending in "/" (page filenames are appended to it).

        Args:
            value (str): The `root_site` value.

        Returns:
            str: `value`, unchanged.

        Raises:
            ValueError: If it isn't an http(s) URL with a host, or doesn't end with "/".
        """
        url = urllib.parse.urlsplit(value)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError(f"root_site must be an http(s) URL with a host: {value!r}")
        if not value.endswith("/"):
            raise ValueError(f'root_site must end with "/": {value!r}')
        return value

    @field_validator("filename_variants")
    @classmethod
    def _check_filename_variants(cls, value: list[str]) -> list[str]:
        """Keep each variant inside a filename, so pages are saved in the download folder.

        Args:
            value (list[str]): The `filename_variants` value.

        Returns:
            list[str]: `value`, unchanged.

        Raises:
            ValueError: If a variant is empty or contains a path separator.
        """
        for variant in value:
            if not variant or any(sep in variant for sep in _PATH_SEPARATORS):
                raise ValueError(
                    f"each filename_variants entry must be a non-empty filename part: {variant!r}"
                )
        return value

    def _under_data(self, *parts: str) -> pathlib.Path:
        """Build a path inside `folder_data`.

        Args:
            *parts (str): Path segments under it.

        Returns:
            pathlib.Path: The joined path.
        """
        return pathlib.Path(self.folder_data).joinpath(*parts)

    @property
    def dl_dir(self) -> pathlib.Path:
        """Directory downloaded pages for `bd_name` are saved to."""
        return self._under_data(self.folder_save_dl, self.bd_name)

    @property
    def upscale_dir(self) -> pathlib.Path:
        """Directory upscaled pages for `bd_name` are saved to."""
        return self._under_data(self.folder_save_upscale, self.bd_name)

    @property
    def models_dir(self) -> pathlib.Path:
        """Directory super-resolution models are stored in."""
        return self._under_data(self.folder_models)


def load_config(path: "str | pathlib.Path") -> AppConfig:
    """Read and validate a `config.json` file.

    Args:
        path: Filesystem path to a JSON object matching `AppConfig`'s fields.

    Returns:
        AppConfig: The validated configuration.

    Raises:
        FileNotFoundError: If `path` does not point to a file.
        json.JSONDecodeError: If the file is not valid JSON.
        pydantic.ValidationError: If the file's content does not match `AppConfig`.
    """
    config_path = pathlib.Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path} (run the command from the folder holding "
            + "config.json, or pass its path with -c/--configuration)"
        )

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw_config)
