#!/usr/bin/env python3
"""Manifest of downloadable OpenCV super-resolution models and the fetch logic for them.

Model files are large (tens of MB) pre-trained TensorFlow weights, hosted on the same GitHub
repositories linked from the project's README. Rather than requiring a manual download, a
model is fetched into `data/models/` the first time it is requested and reused afterwards.
"""

import hashlib
import logging
import pathlib

import requests

from curios_comic_crawler._http import CHUNK_SIZE_BYTES, stream_to_file
from curios_comic_crawler.models import ModelName, ModelSpec

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30

_EDSR_BASE = 'https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models'
_ESPCN_BASE = 'https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export'
_FSRCNN_BASE = 'https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models'
_LAPSRN_BASE = 'https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export'


def _spec(  # noqa: PLR0913
    name: ModelName, scale: int, filename: str, url: str, algorithm: str, sha256: str,
) -> ModelSpec:
    return ModelSpec(
        name=name, scale=scale, filename=filename, url=url,
        algorithm=algorithm,  # pyright: ignore[reportArgumentType]
        sha256=sha256,
    )


MODEL_MANIFEST: dict[tuple[ModelName, int], ModelSpec] = {
    ('edsr', 2): _spec(
        'edsr', 2, 'EDSR_x2.pb', f'{_EDSR_BASE}/EDSR_x2.pb', 'edsr',
        '585623221baa070279a0d1e7e113a4c3faba0f318ca7fdd9a65d9afc0763d9b4',
    ),
    ('edsr', 3): _spec(
        'edsr', 3, 'EDSR_x3.pb', f'{_EDSR_BASE}/EDSR_x3.pb', 'edsr',
        '3baa3740fdb8ee9c52f1a41d69fa74cb9feef0fa9bfeec24f0ee58b928068e9a',
    ),
    ('edsr', 4): _spec(
        'edsr', 4, 'EDSR_x4.pb', f'{_EDSR_BASE}/EDSR_x4.pb', 'edsr',
        'dd35ce3cae53ecee2d16045e08a932c3e7242d641bb65cb971d123e06904347f',
    ),
    ('espcn', 2): _spec(
        'espcn', 2, 'ESPCN_x2.pb', f'{_ESPCN_BASE}/ESPCN_x2.pb', 'espcn',
        '59f77351e1d7c0057bf6fe088b4a8a07e42c468c8c8aebb674a6b4ea1823221d',
    ),
    ('espcn', 3): _spec(
        'espcn', 3, 'ESPCN_x3.pb', f'{_ESPCN_BASE}/ESPCN_x3.pb', 'espcn',
        '0e667db7a431d14c32568ed79bf21701a64df8b1d162e48ca1f8800affe2c2b3',
    ),
    ('espcn', 4): _spec(
        'espcn', 4, 'ESPCN_x4.pb', f'{_ESPCN_BASE}/ESPCN_x4.pb', 'espcn',
        'e403f06309229cf36009cd8fb0da032ba7643fae9f15cf94fe562e8edf8fef47',
    ),
    ('fsrcnn', 2): _spec(
        'fsrcnn', 2, 'FSRCNN_x2.pb', f'{_FSRCNN_BASE}/FSRCNN_x2.pb', 'fsrcnn',
        '366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c',
    ),
    ('fsrcnn', 3): _spec(
        'fsrcnn', 3, 'FSRCNN_x3.pb', f'{_FSRCNN_BASE}/FSRCNN_x3.pb', 'fsrcnn',
        'efd38655a815908c6c8954db6052f128e76a735f1de657894c477d0dc0b64481',
    ),
    ('fsrcnn', 4): _spec(
        'fsrcnn', 4, 'FSRCNN_x4.pb', f'{_FSRCNN_BASE}/FSRCNN_x4.pb', 'fsrcnn',
        '5c68d18db561aed8ead4ffedf1b897ea615baaf60ebf6c35f8e641f8fa4a21bf',
    ),
    ('fsrcnn-small', 2): _spec(
        'fsrcnn-small', 2, 'FSRCNN-small_x2.pb', f'{_FSRCNN_BASE}/FSRCNN-small_x2.pb', 'fsrcnn',
        '429e4793d049c1ae16ddbbc322fd11c3c08831c0c20137390b4d098976a2b0d9',
    ),
    ('fsrcnn-small', 3): _spec(
        'fsrcnn-small', 3, 'FSRCNN-small_x3.pb', f'{_FSRCNN_BASE}/FSRCNN-small_x3.pb', 'fsrcnn',
        '58e72d0c4231b8e754180e787d7348f45ae6230ee3c5af7dc12b558467f593f7',
    ),
    ('fsrcnn-small', 4): _spec(
        'fsrcnn-small', 4, 'FSRCNN-small_x4.pb', f'{_FSRCNN_BASE}/FSRCNN-small_x4.pb', 'fsrcnn',
        '92a9f57c28494813b4dc073b51a300cb909401454d77541bcdc96b4d55901d02',
    ),
    ('lapsrn', 2): _spec(
        'lapsrn', 2, 'LapSRN_x2.pb', f'{_LAPSRN_BASE}/LapSRN_x2.pb', 'lapsrn',
        'f59c86e6835bbca646dbd81588f07820dfe3bd09e3702976ae2d90b5fc0f0b21',
    ),
    ('lapsrn', 4): _spec(
        'lapsrn', 4, 'LapSRN_x4.pb', f'{_LAPSRN_BASE}/LapSRN_x4.pb', 'lapsrn',
        'd3e95c93cafae5ce5a8ed57ce9abf07f2de58da8c5d6d656b766774969835ee2',
    ),
    ('lapsrn', 8): _spec(
        'lapsrn', 8, 'LapSRN_x8.pb', f'{_LAPSRN_BASE}/LapSRN_x8.pb', 'lapsrn',
        'a00211938b2b4c7d23691525a9267d9d0bbb345cdb9f38a7fc917414a5a1884f',
    ),
}


def _sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as model_file:
        for chunk in iter(lambda: model_file.read(CHUNK_SIZE_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(models_dir: pathlib.Path, model_name: ModelName, model_scale: int) -> pathlib.Path:
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
        if _sha256_of(target_path) == spec.sha256:
            return target_path
        logger.warning('%s failed integrity check, re-downloading', target_path)
        target_path.unlink()

    logger.info('Model %s (x%s) not found locally, downloading from %s', spec.name, spec.scale, spec.url)
    models_dir.mkdir(parents=True, exist_ok=True)

    partial_path = target_path.with_suffix(f'{target_path.suffix}.part')
    with requests.get(spec.url, stream=True, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        stream_to_file(response, partial_path, CHUNK_SIZE_BYTES)

    digest = _sha256_of(partial_path)
    if digest != spec.sha256:
        partial_path.unlink(missing_ok=True)
        raise ValueError(
            f'Downloaded {spec.filename} failed integrity check: expected sha256 {spec.sha256}, got {digest}',
        )

    # Atomic rename: a crash mid-download leaves a `.part` file, never a corrupt `.pb` that
    # looks valid to the `is_file()` check above on the next run.
    partial_path.replace(target_path)
    logger.info('Saved model to %s', target_path)

    return target_path
