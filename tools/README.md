# Tools

Maintainer-only scripts. Not part of the installed package.

## `convert_onnx_model.py`

Converts an official xinntao/Real-ESRGAN PyTorch model to ONNX (downloading the weights,
verifying their sha256, exporting with dynamic height/width axes, and sanity-checking the
result), for the `onnx` upscale engine (`sr_engine_onnx.py`).

```sh
pip install -e ".[convert]"
python tools/convert_onnx_model.py
```

Prompts for which model to convert (like `comiccrawler`'s own menu), or skip the prompt with
`--model <key>`:

| `--model`                     | Architecture   | Bundled + wired into `config.json`? |
|--------------------------------|----------------|----------------------------------------|
| `realesr-animevideov3-x4`     | SRVGGNetCompact | Yes -- the fast, anime-tuned default |
| `realesrgan-x4plus-anime-6b`  | RRDBNet (6 blocks) | Yes -- heavier, potentially higher quality |
| `realesr-general-x4v3`        | SRVGGNetCompact | No -- convertible, but not currently an `OnnxModelName` choice |
| `realesrgan-x4plus`           | RRDBNet (23 blocks) | No -- general-purpose (photo), heaviest/slowest, not currently wired in |

By default it overwrites that model's slot in `src/curios_comic_crawler/assets/`; pass
`--output <path>` to write elsewhere instead (e.g. to try a model before committing to it).

Converting one of the "No" rows above does **not** by itself make it usable from
`config.json` -- it only produces the `.onnx` file. Making it a real `upscaler.onnx_model`
choice means also adding it to `models.py`'s `OnnxModelName` and `sr_engine_onnx.py`'s
`_MODEL_FILENAMES`, matching the two models that already are.

Both architectures (`_archs.py`) were verified this session by loading each official checkpoint
with `strict=True` (every layer name/shape must match exactly) and by visually comparing real
output against the official implementation -- not just assumed correct from reading the source.

You do not need any of this to use the package -- the already-converted default model is
already bundled, and running it only needs `onnxruntime` (a base dependency).
