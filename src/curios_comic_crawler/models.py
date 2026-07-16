#!/usr/bin/env python3
"""Pydantic data models shared across the package."""

from typing import Literal

from pydantic import BaseModel

ModelName = Literal['edsr', 'espcn', 'fsrcnn', 'fsrcnn-small', 'lapsrn']

OnnxModelName = Literal['realesr-animevideov3-x4', 'realesrgan-x4plus-anime-6b']


class ModelSpec(BaseModel):
    """A single downloadable OpenCV super-resolution model.

    Attributes:
        name: Model family, as used in `config.json`'s `model_name`.
        scale: Upscaling factor this weight file was trained for.
        filename: Filename the model is stored under in `data/models/`.
        url: Direct download URL for the `.pb` weight file.
        algorithm: Name passed to `cv2.dnn_superres.DnnSuperResImpl.setModel`. Distinct from
            `name` for `fsrcnn-small`, which shares the `fsrcnn` OpenCV algorithm but uses
            smaller trained weights.
        sha256: Expected sha256 hex digest of the weight file, verified after every download
            and before reusing an already-cached file, so a truncated or corrupted `.pb` is
            never silently fed to OpenCV.
    """

    name: ModelName
    scale: int
    filename: str
    url: str
    algorithm: Literal['edsr', 'espcn', 'fsrcnn', 'lapsrn']
    sha256: str
