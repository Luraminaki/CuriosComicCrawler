# Tools

Maintainer-only scripts. Not part of the installed package.

## `convert_onnx_model.py`

Converts an official xinntao/Real-ESRGAN PyTorch model to ONNX, for the `onnx` upscale engine
(`sr_engine_onnx.py`):

1. downloads the weights to a temporary folder and checks their sha256 before using them;
2. loads them with `torch.load(..., weights_only=True)` -- the file comes from the network, so
   it is never unpickled as arbitrary objects;
3. exports with dynamic height/width axes, into the temporary folder;
4. sanity-checks the export (it loads in ONNX Runtime and upscales an arbitrary size 4x), and
   only then copies it to the output path -- a failing export never replaces a bundled model;
5. prints the new file's sha256 (update `_EXPECTED_SHA256` in `tests/test_sr_engine_onnx.py` if
   a bundled model changed).

```sh
pip install -e ".[convert]"     # torch>=2.5 (the first with the `dynamo=` export option) and onnx
python tools/convert_onnx_model.py
```

Prompts for which model to convert (like `comiccrawler`'s own menu), or skip the prompt with
`--model <key>`:

| `--model`                     | Architecture   | Size    |
|--------------------------------|----------------|-----------|
| `realesr-animevideov3-x4`     | SRVGGNetCompact | ~2.5 MB, the fast default |
| `realesrgan-x4plus-anime-6b`  | RRDBNet (6 blocks) | ~17 MB, heavier, potentially higher quality |
| `realesr-general-x4v3`        | SRVGGNetCompact | ~4.9 MB, general-purpose, same speed class as the default |
| `realesrgan-x4plus`           | RRDBNet (23 blocks) | ~64 MB, general-purpose (photo), heaviest/slowest |

All four are currently bundled and wired into `config.json`'s `upscaler.onnx_model`. By default
this tool overwrites that model's slot in `src/curios_comic_crawler/assets/`; pass
`--output <path>` to write elsewhere instead (e.g. to try a model before committing to it).

Converting a model that isn't in the table above (if you add one to `_MODELS` in
`convert_onnx_model.py`) does **not** by itself make it usable from `config.json` -- it only
produces the `.onnx` file. Making it a real `upscaler.onnx_model` choice means also adding it to
`models.py`'s `OnnxModelName` and `sr_engine_onnx.py`'s `_MODEL_FILENAMES`.

Both architectures (`_archs.py`) were verified by loading each official checkpoint with
`strict=True` (every layer name/shape must match exactly) and by visually comparing real output
against the official implementation -- not just assumed correct from reading the source.

You do not need any of this to use the package -- all four converted models are already
bundled, and running them only needs `onnxruntime` (a base dependency).
