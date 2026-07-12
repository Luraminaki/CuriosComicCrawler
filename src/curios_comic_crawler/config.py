#!/usr/bin/env python3
"""Typed configuration for CuriosComicCrawler.

Replaces raw `dict` access into a parsed `config.json` with a validated pydantic model.
"""

import json
import pathlib

from pydantic import BaseModel, Field, field_validator, model_validator

from .model_registry import MODEL_MANIFEST
from .models import ModelName

_WINDOWS_RESERVED_NAMES = frozenset({
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
})


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
        model_name: Super-resolution model family to use.
        model_scale: Upscaling factor to use for `model_name`.
        gray_values: Number of posterisation clusters `area_posterise` reduces each image to.
        headers: HTTP headers sent with every download request.
        upscale_workers: Number of pages to upscale in parallel, each in its own process (every
            worker loads its own copy of the model, so this trades RAM for throughput). `None`
            (the default) uses one worker per CPU core.
    """

    root_site: str
    bd_name: str = Field(alias='BD_name')
    padded: int = Field(gt=0)
    fails: int = Field(gt=0)
    filename_variants: list[str] = Field(min_length=1)
    ext: str
    folder_data: str
    folder_save_dl: str
    folder_save_upscale: str
    folder_models: str
    model_name: ModelName
    model_scale: int
    gray_values: int = Field(ge=2, le=255)
    headers: dict[str, str]
    upscale_workers: int | None = Field(default=None, ge=1)

    model_config = {'populate_by_name': True}

    @field_validator('bd_name')
    @classmethod
    def _check_bd_name(cls, value: str) -> str:
        if not value or value in ('.', '..') or any(sep in value for sep in ('/', '\\')):
            raise ValueError(f'BD_name must be a single path segment, not {value!r}')
        if value.upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f'BD_name {value!r} is a reserved Windows device name')
        return value

    @model_validator(mode='after')
    def _check_root_site(self) -> 'AppConfig':
        if not self.root_site.startswith('http'):
            raise ValueError(f'root_site must start with "http": {self.root_site!r}')
        if not self.root_site.endswith('/'):
            raise ValueError(f'root_site must end with "/": {self.root_site!r}')
        return self

    @model_validator(mode='after')
    def _check_model_combo(self) -> 'AppConfig':
        if (self.model_name, self.model_scale) not in MODEL_MANIFEST:
            known_scales = sorted(scale for name, scale in MODEL_MANIFEST if name == self.model_name)
            raise ValueError(
                f'No known model for model_name={self.model_name!r}, model_scale={self.model_scale!r}. '
                f'Known scales for {self.model_name!r}: {known_scales}'
            )
        return self

    def _under_data(self, *parts: str) -> pathlib.Path:
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


def load_config(path: 'str | pathlib.Path') -> AppConfig:
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
        raise FileNotFoundError(f'Config file not found: {config_path}')

    raw_config = json.loads(config_path.read_text(encoding='utf-8'))
    return AppConfig.model_validate(raw_config)
