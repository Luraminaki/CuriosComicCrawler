#!/usr/bin/env python3
"""Convert an official xinntao/Real-ESRGAN PyTorch model to ONNX, with dynamic H/W axes.

Requires the `convert` extra: `pip install -e ".[convert]"`. Not needed to use the package --
`onnxruntime` (a base dependency) is all that's needed to run an already-converted, bundled
model.

Run with no arguments for an interactive menu (like `comiccrawler`'s own launcher), or pass
`--model <key>` to skip it for scripted use:

    python tools/convert_onnx_model.py
    python tools/convert_onnx_model.py --model realesr-animevideov3-x4

The official conversion script (`Real-ESRGAN/scripts/pytorch2onnx.py`) exports a fixed 64x64
input with no dynamic axes, which isn't usable on real page-sized images -- this adds dynamic
height/width axes instead, so it can't just reuse that script verbatim.
"""

import argparse
import dataclasses
import hashlib
import pathlib
import sys
import tempfile
from collections.abc import Callable

import numpy as np
import onnxruntime
import requests
import torch
from _archs import RRDBNet, SRVGGNetCompact
from torch import nn

_ASSETS_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / 'src' / 'curios_comic_crawler' / 'assets'
)


@dataclasses.dataclass(frozen=True)
class _ModelSpec:
    """One convertible model: where to get it, how to build its architecture, and where it goes.

    Attributes:
        key: Menu/`--model` identifier.
        description: Shown next to `key` in the interactive menu.
        url: Direct download URL for the official `.pth` weights.
        sha256: Expected sha256 hex digest of the downloaded weights, checked before trusting
            them.
        filename: Output filename, relative to `_ASSETS_DIR` when no `--output` is given.
        build_model: Returns a fresh, uninitialized architecture instance ready for
            `load_state_dict`.
    """

    key: str
    description: str
    url: str
    sha256: str
    filename: str
    build_model: Callable[[], nn.Module]


_MODELS: dict[str, _ModelSpec] = {
    spec.key: spec
    for spec in (
        _ModelSpec(
            key='realesr-animevideov3-x4',
            description='SRVGGNetCompact, anime-tuned, 4x -- currently bundled, the fast default',
            url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',
            sha256='b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d',
            filename='realesr-animevideov3.onnx',
            build_model=lambda: SRVGGNetCompact(num_conv=16, upscale=4),
        ),
        _ModelSpec(
            key='realesr-general-x4v3',
            description='SRVGGNetCompact, general-purpose, 4x',
            url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            sha256='8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
            filename='realesr-general-x4v3.onnx',
            build_model=lambda: SRVGGNetCompact(num_conv=32, upscale=4),
        ),
        _ModelSpec(
            key='realesrgan-x4plus-anime-6b',
            description='RRDBNet, anime-tuned (6 blocks), 4x -- heavier, higher quality than animevideov3',
            url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
            sha256='f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da',
            filename='realesrgan-x4plus-anime-6b.onnx',
            build_model=lambda: RRDBNet(scale=4, num_block=6),
        ),
        _ModelSpec(
            key='realesrgan-x4plus',
            description='RRDBNet, general-purpose (photo), 4x -- the heaviest, slowest option',
            url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            sha256='4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
            filename='realesrgan-x4plus.onnx',
            build_model=lambda: RRDBNet(scale=4, num_block=23),
        ),
    )
}


def _prompt_model() -> str:
    """Ask the user which model to convert, looping until a valid number is entered.

    Returns:
        str: The chosen model's `_ModelSpec.key`.
    """
    options = '\n'.join(f'  {i}) {spec.key} -- {spec.description}' for i, spec in enumerate(_MODELS.values(), 1))
    keys_by_number = {str(i): spec.key for i, spec in enumerate(_MODELS.values(), 1)}

    while True:
        answer = input(f'Which model do you want to convert?\n{options}\n> ').strip()
        if answer in keys_by_number:
            return keys_by_number[answer]
        print(f'{answer!r} is not a valid choice.')


def _download_weights(spec: _ModelSpec, dest: pathlib.Path) -> None:
    """Download `spec`'s official weights to `dest`, refusing to keep them if the hash is wrong.

    Args:
        spec (_ModelSpec): Which model to download.
        dest (pathlib.Path): Where to write the downloaded `.pth` file.

    Raises:
        ValueError: If the downloaded content's sha256 doesn't match `spec.sha256`.
    """
    print(f'Downloading {spec.url}')
    response = requests.get(spec.url, timeout=30)
    response.raise_for_status()

    digest = hashlib.sha256(response.content).hexdigest()
    if digest != spec.sha256:
        raise ValueError(f'Downloaded weights failed integrity check: expected sha256 {spec.sha256}, got {digest}')

    dest.write_bytes(response.content)


def _verify(onnx_path: pathlib.Path) -> None:
    """Sanity-check the exported graph: does it load, and does it 4x an arbitrary input size?

    Args:
        onnx_path (pathlib.Path): Path to the exported `.onnx` file.

    Raises:
        ValueError: If the output shape isn't exactly 4x the input's height/width.
    """
    session = onnxruntime.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    dummy = np.random.default_rng(0).random((1, 3, 32, 48), dtype=np.float32)
    (output,) = session.run(None, {'input': dummy})

    expected_shape = (1, 3, 32 * 4, 48 * 4)
    if output.shape != expected_shape:
        raise ValueError(f'Unexpected output shape: {output.shape}, expected {expected_shape}')

    print(f'Verified: input {dummy.shape} -> output {output.shape}')


def convert(spec: _ModelSpec, output_path: pathlib.Path) -> None:
    """Download the official weights, export to ONNX with dynamic H/W, and sanity-check it.

    Args:
        spec (_ModelSpec): Which model to convert.
        output_path (pathlib.Path): Where to write the resulting `.onnx` file.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = pathlib.Path(tmp_dir) / pathlib.Path(spec.url).name
        _download_weights(spec, weights_path)

        model = spec.build_model()
        state_dict = torch.load(weights_path, map_location='cpu')
        # Matches RealESRGANer's own loading logic: prefer the EMA (exponential moving average)
        # weights when present, since that's what every official inference path uses.
        key = 'params_ema' if 'params_ema' in state_dict else 'params'
        model.load_state_dict(state_dict[key])
        model.eval()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.rand(1, 3, 64, 64)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 2: 'height', 3: 'width'},
            },
            dynamo=False,
        )

    print(f'Exported to {output_path}')
    _verify(output_path)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f'sha256: {digest}')
    print('If this is the bundled model and the hash changed, update _EXPECTED_SHA256 in '
          'tests/test_sr_engine_onnx.py.')


def main() -> int:
    """Parse CLI args (or prompt interactively) and run the conversion.

    Returns:
        int: Process exit code (always `0`; failures raise instead).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model', choices=sorted(_MODELS), help='Skip the interactive menu and convert this model directly.',
    )
    parser.add_argument(
        '--output', type=pathlib.Path,
        help='Where to write the .onnx file. Defaults to this model\'s slot in assets/.',
    )
    args = parser.parse_args()

    model_key = args.model or _prompt_model()
    spec = _MODELS[model_key]
    output_path = args.output or (_ASSETS_DIR / spec.filename)

    convert(spec, output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
