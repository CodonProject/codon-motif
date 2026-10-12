# ONNX export

One entry point that turns a MotifChord model into the ONNX bundle the runtime in
`runtime/` serves: a decode graph, one prefill graph per chunk size, and a single
shared weights file.

```
out/
  decode.onnx       input_ids[1, 1]   + position + 56 caches -> logits[1, 1, 40960]    + 56 caches
  prefill_64.onnx   input_ids[1, 64]  + position + 56 caches -> logits[1, 64, 40960]   + 56 caches
  prefill_256.onnx  one graph per requested chunk size - 64, 256, 512, 1024 by default
  weights.bin       the initializers of every graph (byte-identical, written once)
  export.json       precision, chunk sizes, node counts, hashes, sizes, verification
```

A runtime feeds a prompt through the largest chunk that still fits, then through
smaller ones, and decodes the tail: four chords cost four graphs but one weights
file, and a long prompt stops paying per-token decode for its prefix.

```
pip install codon-motif[onnx]
```

## Entry point

```python
from motif.onnx import export_onnx

export_onnx('D:/MotifChord/pretrained_base.safetensors', 'fp16', '.research/onnx/out/fp16')
export_onnx(model, 'fp32', 'out/fp32', prefill_lengths=(128, 512), verify_steps=16)
```

```python
def export_onnx(source, precision='fp16', out_dir=None, *,
                prefill_lengths=(64, 256, 512, 1024),
                verify=True, verify_steps=8, verify_tokens=8, seed=0, progress=True,
                threads=None, solve_patch=True, config=None) -> ExportResult
```

**`source`** is a `MotifChordModel` instance or a checkpoint file / directory. A
directory is searched in this order, because `motif.model.checkpoint` sends every
directory down its safetensors branch and would miss a `last.pt`:

1. `model.safetensors`  2. any other `*.safetensors`  3. `*.index.json`
4. `model.pt` / `last.pt` / `last.pth` / `pytorch_model.bin`  5. the newest other
weight file

**`precision`** is `'fp32'` or `'fp16'`. The graphs are traced from a model of that
dtype (`fp16` deep-copies and casts the loaded model), so precision is chosen the
same way as the weights to serve. `fp16` halves the bundle.

**`prefill_lengths`** is one chunk size or a sequence of them, each becoming a
`prefill_<T>.onnx` (default `(64, 256, 512, 1024)`; `prefill_lengths=256` exports
just that one, and the CLI takes `--prefill-lengths 64,256` or `--prefill-length
64`). `T` is
static in the graph - the chunked delta rule bakes its boundaries (64) in - so a
runtime that wants to prefill fast at several prompt lengths needs one graph per
size. The graphs share their initializers, so an extra chunk size costs graph
structure, not weights.

**`out_dir`** is created when missing; a previous bundle in it (`decode.onnx`,
`prefill_<T>.onnx`, `prefill.onnx`, `weights.bin`, `export.json`, `decode/`,
`prefill*`) is replaced, so re-exporting with different chunk sizes leaves no
stale graphs behind.

Durations on the real 172 M checkpoint: ~34 s to load, then per graph roughly `4.7 s
+ 13.8 s` (decode), `17.3 s + 40.8 s` (T=64), `18.2 s + 46.0 s` (T=512) for
`torch.export` + `torch.onnx.export` - four minutes forty-four in total, most of it
translating graph structure, not weights.

## Progress bar

```
onnx export | [######..............]  29% 5/10 export prefill 256  61.2s
```

Every phase is weighted up front - each prefill graph by its chunk size, since
trace, serialization and hashing all grow with `T` - so the percentage is real
work. A background ticker keeps the line alive while `torch.export` /
`torch.onnx.export` / `onnx.save_model` block (they print nothing for minutes), and
each finished phase leaves a line behind:

```
[onnx export] 1/10 resolve source  0.0s  (  2%)
[onnx export] 2/10 load weights  33.9s  (  8%)
[onnx export] 3/10 prepare  2.3s  ( 10%)
[onnx export] 4/10 export decode  13.8s  ( 17%)
```

`verify` counts its units (decode steps, then every prefill chunk) and shows them.
When stderr is not a terminal - a log file, CI - the bar degrades to one line per
phase, which is what `.tmp/*.log` runs show. `--no-progress` (or
`progress=False`) prints only the notes: source, model shape, hashes, results.

## Files

| File | Role |
|---|---|
| `export_onnx.py` | entry point + CLI + bundle writing + ORT/eager verification |
| `step.py` | `ModelStep`: the exportable `(ids, position, caches) -> (logits, caches)` signature, flat state packing over the library's wide functional cache |
| `reader.py` | `resolve_source` / `load_model`: model instance or weights directory → `MotifChordModel` |
| `solve.py` | row-forward triangle inverse swapped in for `torch.linalg.solve_triangular` (no ONNX function exists for it; E4 measured 1.955e-07) |
| `progress.py` | the bar (no torch / onnx import) |
| `runtime/` | the torch-free executor the service and the research CLIs use |

## CLI

```powershell
python -m motif.onnx.export_onnx `
  --weights D:\MotifChord\pretrained_base.safetensors `
  --precision fp16 `
  --out .research\onnx\out\fp16
```

| Flag | Default | Meaning |
|---|---|---|
| `--weights` / `--source` | required | checkpoint file or directory |
| `--precision` | `fp16` | `fp32` or `fp16` |
| `--out` | required | destination directory |
| `--prefill-lengths` | `64,256,512,1024` | chunk sizes of the prefill graphs |
| `--prefill-length` | | one chunk size, shorthand for `--prefill-lengths N` |
| `--model-config` | | config JSON for a checkpoint that carries none |
| `--no-verify` | | skip the onnxruntime comparison |
| `--verify-steps` / `--verify-tokens` | `8` / `8` | decode steps compared, tokens decoded to warm the cache |
| `--seed` | `0` | seeds the ids used for tracing and verifying |
| `--threads` | | onnxruntime intra-op threads while verifying |
| `--no-progress` | | notes only |
| `--keep-solve` | | do not patch `torch.linalg.solve_triangular` |

## Verification

`verify` (default) runs the bundled graphs under `onnxruntime` with the CPU
provider and compares them with the eager model **of the same precision**:

* `decode.onnx`: the cache is warmed for `verify_tokens`, then `verify_steps`
  single-token steps are compared position by position;
* every `prefill_<T>.onnx`: one chunk of `T` tokens from an empty cache is
  compared position by position (this one also exercises the empty-sequence
  case, where the chord cache has length 0).

Each reports worst absolute logit difference and argmax agreement; `export.json`
keeps them. A correct bundle prints `argmax 100.000%` and a worst difference of
about `1.6e-01` in fp16 and `2e-05` in fp32 (the delta rule accumulates over the
sequence, so fp16 drifts more). On a small untrained model the fp16 argmax
agreement can dip - flat logits mean a `1e-2` rounding difference flips
near-ties - while the fp32 graph of the same model still agrees 100%. Position 0
and position 1 agreeing is what rules out an offset.

## Measured

`D:\MotifChord\pretrained_base.safetensors` at `fp16`, 172 M parameters, default
chunk sizes, 4m43.7s:

| | |
|---|---|
| nodes | decode 2764, prefill 64 / 256 / 512 / 1024 = 2996 / 3752 / 4760 / 6776 |
| `weights.bin` | 328.5 MiB, `sha256[:16] = 5acb33190436412a`, shared by all five graphs (separate blobs would cost 1642.7 MiB) |
| bundle | 5.9 + 23.0 + 24.8 + 27.4 + 32.5 + 328.5 = 442.1 MiB |
| decode | worst 1.680e-01, argmax 100.000% (8/8), 30.2 ms per step |
| prefill 64 | worst 1.562e-01, argmax 100.000% (64/64), 305.2 ms |
| prefill 256 | worst 2.197e-01, argmax 100.000% (256/256), 928.7 ms |
| prefill 512 | worst 4.453e-01, argmax 99.414% (509/512), 2252.9 ms |
| prefill 1024 | worst 4.063e-01, argmax 99.512% (1019/1024), 6248.9 ms |

Longer chunks drift a little more in fp16 (the delta rule accumulates along the
sequence) - 512 and 1024 flip a handful of near-ties, exactly as an fp16 run of
the same length does.

The decode graph is interchangeable with the one the F-series experiments
produced (same initializer hash, same node count). `motif.service` serves this
bundle, and the JavaScript runtime under `.research/runtime/js` serves it too
(`check_runtime.mjs --engine`, `cli.mjs --bundle <dir>`).

## Notes

* The cached sequence length stays **dynamic** (`torch.export.Dim('S')`), so one
  decode graph serves every context length. The token count `T` is static: the
  chunked delta rule bakes its chunk boundaries (64) in, and `torch.export`
  cannot express "a multiple of 64" as a symbolic dimension.
* Chord keys / values are stored already expanded to `num_heads` in the graph.
  `expand` / `repeat_interleave` / broadcast GQA on the dynamic axis trips
  `Constraints violated (S)` in `torch.export`; widening once at update time
  avoids it and costs 4x KV memory (~100 MB at 4096 tokens, 4 chord layers).
  This is a library switch, not a tool trick: the export calls
  `model.create_cache(functional=True, wide_kv=True)` (which is
  `motif.model.cache.build_cache`), so tracing gets an out-of-place cache that
  writes one copy per head, and a checkpoint saved by an earlier revision loads
  unchanged (`tests/test_unified_cache.py` covers both the numerics and the
  dynamic-length export).
* Trace-time example state uses `TRACE_CACHE_LENGTH = 16` key/value rows, and the
  `position` input is a tensor, so neither the cache length nor the position is
  frozen into the graph.
* One `weights.bin` serves every graph **on disk**, but a runtime pays per session:
  onnxruntime copies the initializers into each one, so four resident sessions are
  ~1.4 GB. The Python runtime holds all of them; the wasm build (browser / Node)
  does not, which is why `src/engine.js` creates chunk sessions lazily and drops a
  chunk that fails. There, `prefill_1024.onnx` is refused with `std::bad_alloc`
  (weights plus a whole 1024-token chunk of activations in one 32-bit heap) and the
  prompt is served in 512-token pieces; native onnxruntime runs it in 6.2 s.
* `motif.onnx` imports `onnx`, `onnxruntime` and `numpy`, which is why they sit in
  the `onnx` extra rather than in the library's core dependencies. Nothing under
  `motif.onnx.runtime` imports `torch`.
