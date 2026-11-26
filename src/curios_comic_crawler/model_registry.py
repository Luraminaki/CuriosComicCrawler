#!/usr/bin/env python3
"""Manifest of downloadable OpenCV super-resolution models and the fetch logic for them.

Model files are large (tens of MB) pre-trained TensorFlow weights, hosted on the same GitHub
repositories linked from the project's README. Rather than requiring a manual download, a
model is fetched into `data/models/` the first time it is requested and reused afterwards.
"""

import logging
from typing import TYPE_CHECKING

from curios_comic_crawler._http import download_verified, sha256_of
from curios_comic_crawler.models import ModelSpec

if TYPE_CHECKING:
    import pathlib

    from curios_comic_crawler.models import ModelName

logger = logging.getLogger(__name__)


_EDSR_BASE = "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models"
_ESPCN_BASE = "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export"
_FSRCNN_BASE = "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models"
_LAPSRN_BASE = "https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export"


_SPECS = (
    ModelSpec(
        name="edsr",
        scale=2,
        filename="EDSR_x2.pb",
        url=f"{_EDSR_BASE}/EDSR_x2.pb",
        algorithm="edsr",
        sha256="585623221baa070279a0d1e7e113a4c3faba0f318ca7fdd9a65d9afc0763d9b4",
    ),
    ModelSpec(
        name="edsr",
        scale=3,
        filename="EDSR_x3.pb",
        url=f"{_EDSR_BASE}/EDSR_x3.pb",
        algorithm="edsr",
        sha256="3baa3740fdb8ee9c52f1a41d69fa74cb9feef0fa9bfeec24f0ee58b928068e9a",
    ),
    ModelSpec(
        name="edsr",
        scale=4,
        filename="EDSR_x4.pb",
        url=f"{_EDSR_BASE}/EDSR_x4.pb",
        algorithm="edsr",
        sha256="dd35ce3cae53ecee2d16045e08a932c3e7242d641bb65cb971d123e06904347f",
    ),
    ModelSpec(
        name="espcn",
        scale=2,
        filename="ESPCN_x2.pb",
        url=f"{_ESPCN_BASE}/ESPCN_x2.pb",
        algorithm="espcn",
        sha256="59f77351e1d7c0057bf6fe088b4a8a07e42c468c8c8aebb674a6b4ea1823221d",
    ),
    ModelSpec(
        name="espcn",
        scale=3,
        filename="ESPCN_x3.pb",
        url=f"{_ESPCN_BASE}/ESPCN_x3.pb",
        algorithm="espcn",
        sha256="0e667db7a431d14c32568ed79bf21701a64df8b1d162e48ca1f8800affe2c2b3",
    ),
    ModelSpec(
        name="espcn",
        scale=4,
        filename="ESPCN_x4.pb",
        url=f"{_ESPCN_BASE}/ESPCN_x4.pb",
        algorithm="espcn",
        sha256="e403f06309229cf36009cd8fb0da032ba7643fae9f15cf94fe562e8edf8fef47",
    ),
    ModelSpec(
        name="fsrcnn",
        scale=2,
        filename="FSRCNN_x2.pb",
        url=f"{_FSRCNN_BASE}/FSRCNN_x2.pb",
        algorithm="fsrcnn",
        sha256="366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c",
    ),
    ModelSpec(
        name="fsrcnn",
        scale=3,
        filename="FSRCNN_x3.pb",
        url=f"{_FSRCNN_BASE}/FSRCNN_x3.pb",
        algorithm="fsrcnn",
        sha256="efd38655a815908c6c8954db6052f128e76a735f1de657894c477d0dc0b64481",
    ),
    ModelSpec(
        name="fsrcnn",
        scale=4,
        filename="FSRCNN_x4.pb",
        url=f"{_FSRCNN_BASE}/FSRCNN_x4.pb",
        algorithm="fsrcnn",
        sha256="5c68d18db561aed8ead4ffedf1b897ea615baaf60ebf6c35f8e641f8fa4a21bf",
    ),
    ModelSpec(
        name="fsrcnn-small",
        scale=2,
        filename="FSRCNN-small_x2.pb",
        url=f"{_FSRCNN_BASE}/FSRCNN-small_x2.pb",
        algorithm="fsrcnn",
        sha256="429e4793d049c1ae16ddbbc322fd11c3c08831c0c20137390b4d098976a2b0d9",
    ),
    ModelSpec(
        name="fsrcnn-small",
        scale=3,
        filename="FSRCNN-small_x3.pb",
        url=f"{_FSRCNN_BASE}/FSRCNN-small_x3.pb",
        algorithm="fsrcnn",
        sha256="58e72d0c4231b8e754180e787d7348f45ae6230ee3c5af7dc12b558467f593f7",
    ),
    ModelSpec(
        name="fsrcnn-small",
        scale=4,
        filename="FSRCNN-small_x4.pb",
        url=f"{_FSRCNN_BASE}/FSRCNN-small_x4.pb",
        algorithm="fsrcnn",
        sha256="92a9f57c28494813b4dc073b51a300cb909401454d77541bcdc96b4d55901d02",
    ),
    ModelSpec(
        name="lapsrn",
        scale=2,
        filename="LapSRN_x2.pb",
        url=f"{_LAPSRN_BASE}/LapSRN_x2.pb",
        algorithm="lapsrn",
        sha256="f59c86e6835bbca646dbd81588f07820dfe3bd09e3702976ae2d90b5fc0f0b21",
    ),
    ModelSpec(
        name="lapsrn",
        scale=4,
        filename="LapSRN_x4.pb",
        url=f"{_LAPSRN_BASE}/LapSRN_x4.pb",
        algorithm="lapsrn",
        sha256="d3e95c93cafae5ce5a8ed57ce9abf07f2de58da8c5d6d656b766774969835ee2",
    ),
    ModelSpec(
        name="lapsrn",
        scale=8,
        filename="LapSRN_x8.pb",
        url=f"{_LAPSRN_BASE}/LapSRN_x8.pb",
        algorithm="lapsrn",
        sha256="a00211938b2b4c7d23691525a9267d9d0bbb345cdb9f38a7fc917414a5a1884f",
    ),
)
# Keyed by (name, scale), the pair a config names a model by.
MODEL_MANIFEST: dict[tuple["ModelName", int], ModelSpec] = {
    (spec.name, spec.scale): spec for spec in _SPECS
}


def ensure_model(
    models_dir: "pathlib.Path", model_name: "ModelName", model_scale: int
) -> "pathlib.Path":
    """Return the local path to a model, downloading it into `models_dir` if missing.

    A cached file that already exists locally is still hashed and checked against the
    manifest every call: cheap relative to the multi-second upscale it precedes, and it
    catches a `.pb` that got corrupted or truncated after being saved (disk issue, an
    interrupted manual copy, etc.) instead of silently handing it to OpenCV.

    Args:
        models_dir (pathlib.Path): Directory models are stored in and fetched into.
        model_name (ModelName): Model family (e.g. `"edsr"`).
        model_scale (int): Upscaling factor (e.g. `4`).

    Returns:
        pathlib.Path: Path to the (now guaranteed to exist and match) model weight file.

    Raises:
        KeyError: If `(model_name, model_scale)` isn't a known model/scale combination.
        requests.RequestException: If the download fails.
        ValueError: If the downloaded file doesn't match the manifest's sha256.
    """
    spec = MODEL_MANIFEST[(model_name, model_scale)]
    target_path = models_dir / spec.filename

    if target_path.is_file() and target_path.stat().st_size > 0:
        if sha256_of(target_path) == spec.sha256:
            return target_path
        logger.warning("%s failed integrity check, re-downloading", target_path)

    logger.info(
        "Model %s (x%s) not found locally, downloading from %s", spec.name, spec.scale, spec.url
    )
    download_verified(spec.url, target_path, spec.sha256)
    logger.info("Saved model to %s", target_path)
    return target_path
