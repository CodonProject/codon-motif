import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence, Union

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault('TRITON_CACHE_DIR', os.path.join(_REPO, '.tmp', 'triton'))
for _stream in (sys.stdout, sys.stderr):
    _stream.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx.external_data_helper import convert_model_to_external_data

from motif.onnx.progress import Progress, format_seconds
from motif.onnx.reader import load_model, resolve_source
from motif.onnx.solve import install_solve_patch
from motif.onnx.step import (
    ModelStep,
    dynamic_specs,
    flat_state,
    layer_plan,
    new_cache,
    zero_flats,
)

ort.set_default_logger_severity(3)

Precision = Literal['fp32', 'fp16']
PRECISIONS = ('fp32', 'fp16')
TORCH_DTYPES = {'fp32': torch.float32, 'fp16': torch.float16}
DEFAULT_PREFILL_LENGTHS = (64, 256, 512, 1024)
BUNDLE_FILES = ('decode.onnx', 'weights.bin', 'export.json')
TRACE_CACHE_LENGTH = 16


def _phases(prefill_lengths, verify):
    '''
    The weighted phase list, built before the bar exists.

    A prefill graph costs roughly its token count to trace and serialize, and it
    brings a `weights.bin` of its own to hash, so its weight follows the length.
    '''
    phases = [('resolve source', 1.0), ('load weights', 2.0), ('prepare', 1.0),
              ('export decode', 3.0)]
    for length in prefill_lengths:
        phases.append((f'export prefill {length}', 3.0 * length / 64.0))
    phases.append(('write bundle', 2.0 + 0.5 * len(prefill_lengths)))
    if verify:
        phases.append(('verify', 4.0 + sum(length / 256.0 for length in prefill_lengths)))
    return phases


def _prefill_lengths(value):
    '''`64` or `[64, 256]` becomes a sorted tuple of unique positive lengths.'''
    if isinstance(value, int):
        value = (value,)
    lengths = sorted({int(length) for length in value})
    if not lengths:
        raise ValueError('prefill_lengths must hold at least one chunk size.')
    if lengths[0] < 1:
        raise ValueError(f'prefill_lengths must be positive, got {list(lengths)}.')
    return tuple(lengths)


def _prefill_name(length):
    return f'prefill_{length}'


@dataclass
class ExportResult:
    '''What `export_onnx` produced.'''

    out_dir: str
    precision: str
    source: str
    decode: str
    prefills: Dict[int, str]
    weights: str
    weights_sha256: str
    shared_weights: bool
    decode_nodes: int
    prefill_nodes: Dict[int, int] = field(default_factory=dict)
    files: Dict[str, int] = field(default_factory=dict)
    verification: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def manifest(self):
        '''The `export.json` payload.'''
        return {
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'source': self.source,
            'precision': self.precision,
            'decode': {'file': os.path.relpath(self.decode, self.out_dir),
                       'nodes': self.decode_nodes,
                       'bytes': self.files.get(os.path.relpath(self.decode, self.out_dir), 0)},
            'prefills': [{'length': length,
                          'file': os.path.relpath(self.prefills[length], self.out_dir),
                          'nodes': self.prefill_nodes.get(length, 0),
                          'bytes': self.files.get(os.path.relpath(self.prefills[length],
                                                                  self.out_dir), 0)}
                         for length in sorted(self.prefills)],
            'weights': {'file': os.path.relpath(self.weights, self.out_dir),
                        'shared': self.shared_weights,
                        'sha256': self.weights_sha256,
                        'bytes': self.files.get(os.path.relpath(self.weights, self.out_dir), 0)},
            'verification': self.verification,
            'files': self.files,
            'seconds': round(self.seconds, 2),
            'environment': {
                'python': sys.version.split()[0],
                'torch': torch.__version__,
                'onnx': onnx.__version__,
                'onnxruntime': ort.__version__,
            },
        }


def export_onnx(
    source: Union['MotifChordModel', str, os.PathLike],
    precision: Precision = 'fp16',
    out_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    prefill_lengths: Union[int, Sequence[int]] = DEFAULT_PREFILL_LENGTHS,
    verify: bool = True,
    verify_steps: int = 8,
    verify_tokens: int = 8,
    seed: int = 0,
    progress: bool = True,
    threads: Optional[int] = None,
    solve_patch: bool = True,
    config=None,
) -> ExportResult:
    '''
    Export the ONNX decode / prefill bundle of a model.

    Args:
        source (Union[MotifChordModel, str, os.PathLike]): A live model, or a
            checkpoint file / directory (see `reader.resolve_source` for the
            directory order).
        precision (Literal['fp32', 'fp16']): Weight precision of the graphs. The
            graphs are traced from the model itself, so this is the same decision
            as which model to serve; `fp16` halves the bundle.
        out_dir (Optional[Union[str, os.PathLike]]): Destination directory, created
            when missing and cleaned of a previous bundle when not.
        prefill_lengths (Union[int, Sequence[int]]): Static token counts of the
            prefill graphs; one `prefill_<T>.onnx` comes out per value. A runtime
            picks the largest one that still fits, so a wider spread means fewer
            calls and a bigger bundle.
        verify (bool): Run every graph under onnxruntime and compare it with the
            same-precision eager model.
        verify_steps (int): Decode steps compared when verifying.
        verify_tokens (int): Tokens decoded to warm the cache before those steps.
        seed (int): Seeds the random ids used for tracing and verifying.
        progress (bool): Draw the progress bar.
        threads (Optional[int]): onnxruntime intra-op threads; None uses its default.
        solve_patch (bool): Swap in the row-forward triangle inverse the exporter
            needs (see `solve.py`).
        config (Optional[Union[MotifChordConfig, Dict, str]]): Explicit model config
            for checkpoints that carry none.

    Returns:
        ExportResult: Paths of the decode and prefill graphs plus the weights,
        node counts, file sizes, verification numbers and the elapsed time.

    Raises:
        ValueError: If `precision`, `out_dir` or `prefill_lengths` is missing or
            invalid.
    '''
    if precision not in PRECISIONS:
        raise ValueError(
            f'precision must be one of {list(PRECISIONS)}, got {precision!r}; '
            f'the arguments are export_onnx(source, precision, out_dir).')
    if out_dir is None:
        raise ValueError(
            'out_dir is required: export_onnx(source, precision, out_dir).')
    lengths = _prefill_lengths(prefill_lengths)
    started = time.perf_counter()
    out_dir = os.path.abspath(os.fspath(out_dir))
    if solve_patch:
        install_solve_patch()
    torch.manual_seed(seed)
    bar = Progress(_phases(lengths, verify), label='onnx export', enabled=progress)
    verification: Dict[str, float] = {}
    result = None

    with bar:
        with bar.phase('resolve source'):
            path, description = resolve_source(source)
            bar.note(f'[onnx] source {description}')

        with bar.phase('load weights'):
            base = load_model(source, config=config)
            bar.note(f'[onnx] loaded {_describe_model(base)}')

        with bar.phase('prepare'):
            model = base if precision == 'fp32' else copy.deepcopy(base).half()
            if model is not base:
                del base
            model.eval()
            kinds = layer_plan(model)
            generator = torch.Generator().manual_seed(seed + 1)
            length = max(max(lengths), verify_tokens + verify_steps) + 1
            ids = torch.randint(0, model.embed.num_embeddings, (1, length),
                                generator=generator)
            with torch.no_grad():
                probe = model(ids[:, :1], cache=new_cache(model))
            template = flat_state(model, probe.cache)
            start_state = zero_flats(model, template)
            trace_state = [
                tensor.new_zeros(1, tensor.shape[1], TRACE_CACHE_LENGTH, tensor.shape[3])
                if kind == 'kv' else tensor
                for kind, tensor in zip(kinds, start_state)
            ]
            step_module = ModelStep(model)
            specs = dynamic_specs(model)
            bar.note(f'[onnx] {precision} weights {_parameter_bytes(model) / 2**20:.1f} MiB, '
                     f'{len(kinds)} cache tensors')
            bar.note(f'[onnx] {_describe_attention(model)}')
            bar.note('[onnx] initial state ' + ', '.join(
                f'{tuple(t.shape)} {str(t.dtype).replace("torch.", "")}'
                for t in start_state[:3]) + f', ... {len(start_state)} tensors')

        with bar.phase('export decode', units=2) as phase:
            program = _trace(step_module, (ids[:, :1], torch.tensor([0]), tuple(trace_state)),
                             (None, None, specs))
            decode_nodes = _node_count(program)
            phase.advance(1, f'{decode_nodes} nodes ({format_seconds(phase.elapsed)})')
            phase.note(f'[onnx] decode torch.export {decode_nodes} nodes '
                       f'({format_seconds(phase.elapsed)})')
            onnx_decode = _to_onnx(program)
            phase.advance(1, format_seconds(phase.elapsed))
            phase.note(f'[onnx] decode torch.onnx {format_seconds(phase.elapsed)}')

        onnx_prefills: Dict[int, object] = {}
        prefill_nodes: Dict[int, int] = {}
        for length in lengths:
            with bar.phase(f'export prefill {length}', units=2) as phase:
                program = _trace(
                    step_module,
                    (ids[:, :length], torch.tensor([0]), tuple(trace_state)),
                    (None, None, specs))
                prefill_nodes[length] = _node_count(program)
                phase.advance(1, f'{prefill_nodes[length]} nodes ({format_seconds(phase.elapsed)})')
                phase.note(f'[onnx] prefill T={length} torch.export {prefill_nodes[length]} nodes '
                           f'({format_seconds(phase.elapsed)})')
                onnx_prefills[length] = _to_onnx(program)
                phase.advance(1, format_seconds(phase.elapsed))
                phase.note(f'[onnx] prefill T={length} torch.onnx {format_seconds(phase.elapsed)}')

        with bar.phase('write bundle'):
            decode_path, prefill_paths, weights_path, weights_hash, shared = _write_bundle(
                out_dir, onnx_decode, onnx_prefills, lengths, bar)

        if verify:
            with bar.phase('verify', units=verify_steps + len(lengths)) as phase:
                verification = _verify(model, ids, decode_path, prefill_paths, start_state,
                                       verify_tokens, verify_steps, lengths, threads, phase)

        result = ExportResult(
            out_dir=out_dir,
            precision=precision,
            source=path or 'model instance',
            decode=decode_path,
            prefills=prefill_paths,
            weights=weights_path,
            weights_sha256=weights_hash,
            shared_weights=shared,
            decode_nodes=decode_nodes,
            prefill_nodes=prefill_nodes,
            files=_files_of(out_dir),
            verification=verification,
            seconds=time.perf_counter() - started,
        )
        _write_manifest(result)
        nodes = ', '.join(f'prefill {length} {prefill_nodes[length]}' for length in lengths)
        bar.note(f'[onnx] bundle {out_dir} '
                 f'{sum(result.files.values()) / 2**20:.1f} MiB '
                 f'(decode {decode_nodes} nodes, {nodes} nodes, '
                 f'{"shared" if shared else "separate"} weights)')

    if result.files:
        for name in sorted(result.files):
            print(f'[onnx] {name:24s} {result.files[name] / 2**20:8.1f} MiB')
    print(f'[onnx] wrote {os.path.join(out_dir, "export.json")}')
    print(f'[onnx] done in {format_seconds(result.seconds)}')
    return result


def _trace(module, example_inputs, dynamic_shapes):
    '''`torch.export` the step module, keeping the caches dynamic.'''
    return torch.export.export(module, tuple(example_inputs), dynamic_shapes=dynamic_shapes)


def _node_count(program):
    return sum(1 for _ in program.graph_module.graph.nodes)


def _to_onnx(program):
    '''An exported program becomes an ONNX program; weights stay inside the proto.'''
    return torch.onnx.export(program, external_data=False, optimize=False)


def _save_with_weights(program, directory, name):
    '''Write `<name>.onnx` plus a `weights.bin` next to it, every initializer inside.'''
    os.makedirs(directory, exist_ok=True)
    model_proto = program.model_proto
    convert_model_to_external_data(
        model_proto, all_tensors_to_one_file=True, location='weights.bin',
        size_threshold=0, convert_attribute=False)
    path = os.path.join(directory, f'{name}.onnx')
    onnx.save_model(model_proto, path)
    return path


def _write_bundle(out_dir, onnx_decode, onnx_prefills, lengths, bar):
    '''
    Stage every graph, decide whether their initializers are the same bytes, and
    land `decode.onnx` / `prefill_<T>.onnx` / `weights.bin` in `out_dir`.

    The initializers come from one model, so they are byte-identical unless a
    graph folded them differently; the check keeps the common case at one copy.
    '''
    staging = os.path.join(out_dir, '_staging')
    shutil.rmtree(staging, ignore_errors=True)
    blobs = {}
    _save_with_weights(onnx_decode, os.path.join(staging, 'decode'), 'decode')
    blobs['decode'] = os.path.join(staging, 'decode', 'weights.bin')
    for length in lengths:
        name = _prefill_name(length)
        _save_with_weights(onnx_prefills[length], os.path.join(staging, name), name)
        blobs[name] = os.path.join(staging, name, 'weights.bin')
    digests = {name: _digest(blob) for name, blob in blobs.items()}
    rows = {'decode': 'decode', **{_prefill_name(length): f'prefill {length}' for length in lengths}}
    for name, (digest, size) in digests.items():
        bar.note(f'[onnx] {rows[name]:>14} weights.bin {size / 2**20:8.1f} MiB {digest}')
    shared = len(set(digests.values())) == 1
    copies = sum(size for _, size in digests.values())
    bar.note(f'[onnx] byte-identical initializers: {shared} '
             f'(separate blobs would cost {copies / 2**20:.1f} MiB)')

    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(out_dir)):
        target = os.path.join(out_dir, name)
        drop = (name in BUNDLE_FILES or name == 'prefill.onnx'
                or (name.startswith('prefill_') and name.endswith('.onnx')))
        if os.path.isfile(target) and drop:
            os.remove(target)
        elif os.path.isdir(target) and (name == 'decode' or name.startswith('prefill')):
            shutil.rmtree(target, ignore_errors=True)

    if shared:
        decode_path = os.path.join(out_dir, 'decode.onnx')
        weights_path = os.path.join(out_dir, 'weights.bin')
        prefill_paths = {length: os.path.join(out_dir, f'{_prefill_name(length)}.onnx')
                         for length in lengths}
        shutil.move(os.path.join(staging, 'decode', 'decode.onnx'), decode_path)
        for length in lengths:
            name = _prefill_name(length)
            shutil.move(os.path.join(staging, name, f'{name}.onnx'), prefill_paths[length])
        shutil.move(blobs['decode'], weights_path)
    else:
        for name in blobs:
            shutil.move(os.path.join(staging, name), os.path.join(out_dir, name))
        decode_path = os.path.join(out_dir, 'decode', 'decode.onnx')
        weights_path = os.path.join(out_dir, 'decode', 'weights.bin')
        prefill_paths = {length: os.path.join(out_dir, _prefill_name(length),
                                             f'{_prefill_name(length)}.onnx')
                         for length in lengths}
    weights_hash = digests['decode'][0]
    shutil.rmtree(staging, ignore_errors=True)
    return decode_path, prefill_paths, weights_path, weights_hash, shared


def _verify(model, ids, decode_path, prefill_paths, start_state, verify_tokens,
            verify_steps, lengths, threads, phase):
    '''Run every graph under onnxruntime and compare it with eager PyTorch.'''
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = int(threads)
    providers = ['CPUExecutionProvider']
    decode_session = ort.InferenceSession(decode_path, options, providers=providers)
    names = [item.name for item in decode_session.get_inputs()]

    cache = _warm(model, ids, verify_tokens)
    eager_logits, _ = _eager_steps(model, ids, verify_tokens, verify_steps, cache)
    state = [tensor.detach().cpu().numpy()
             for tensor in flat_state(model, _warm(model, ids, verify_tokens))]
    ort_logits = []
    started = time.perf_counter()
    for step in range(verify_steps):
        feeds = {
            'input_ids': ids[:, verify_tokens + step:verify_tokens + step + 1].numpy(),
            'position': np.array([verify_tokens + step], dtype=np.int64),
        }
        feeds.update(dict(zip(names[2:], state)))
        results = decode_session.run(None, feeds)
        ort_logits.append(torch.from_numpy(results[0])[0, 0])
        state = list(results[1:])
        phase.advance(1, f'{step + 1}/{verify_steps} decode steps')
    per_step = (time.perf_counter() - started) / max(1, verify_steps) * 1e3
    worst, rate = _compare('decode.onnx vs eager', ort_logits, eager_logits, phase)

    verification = {
        'decode_worst': worst,
        'decode_argmax': rate,
        'decode_ms_per_step': per_step,
    }
    for length in lengths:
        session = ort.InferenceSession(prefill_paths[length], options, providers=providers)
        p_names = [item.name for item in session.get_inputs()]
        feeds = {
            'input_ids': ids[:, :length].numpy(),
            'position': np.array([0], dtype=np.int64),
        }
        feeds.update(dict(zip(p_names[2:], [t.numpy() for t in start_state])))
        started = time.perf_counter()
        p_results = session.run(None, feeds)
        prefill_ms = (time.perf_counter() - started) * 1e3
        phase.advance(1, f'prefill chunk of {length}')
        with torch.no_grad():
            reference = model(ids[:, :length], cache=model.create_cache(),
                              position=torch.tensor([0]))
        got = [torch.from_numpy(p_results[0])[0, i] for i in range(length)]
        want = [reference.logits[0, i] for i in range(length)]
        p_worst, p_rate = _compare(f'prefill_{length}.onnx vs eager chunk', got, want, phase)
        verification[f'prefill_{length}_worst'] = p_worst
        verification[f'prefill_{length}_argmax'] = p_rate
        verification[f'prefill_{length}_ms'] = prefill_ms
    return verification


def _warm(model, ids, length):
    '''A real cache holding the first `length` tokens, positions `0..length-1`.'''
    cache = model.create_cache(functional=True, wide_kv=True)
    with torch.no_grad():
        for i in range(length):
            out = model(ids[:, i:i + 1], cache=cache, position=torch.tensor([i]))
            cache = out.cache
    return cache


def _eager_steps(model, ids, start, steps, cache):
    logits = []
    with torch.no_grad():
        for i in range(steps):
            out = model(ids[:, start + i:start + i + 1], cache=cache,
                        position=torch.tensor([start + i]))
            logits.append(out.logits[:, 0])
            cache = out.cache
    return logits, cache


def _compare(name, got, want, phase=None):
    '''Worst absolute difference and argmax agreement between two logit lists.'''
    worst = 0.0
    agree = 0
    for a, b in zip(got, want):
        worst = max(worst, (a - b).abs().max().item())
        agree += int(a.argmax().item() == b.argmax().item())
    rate = 100.0 * agree / max(1, len(want))
    line = (f'[onnx] {"ok" if rate == 100.0 else "diff"} {name}: '
            f'worst {worst:.3e} | argmax {rate:.3f}% ({agree}/{len(want)})')
    if phase is not None:
        phase.note(line)
    else:
        print(line)
    return worst, rate


def _write_manifest(result):
    path = os.path.join(result.out_dir, 'export.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(result.manifest(), handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return path


def _files_of(out_dir):
    files = {}
    for root, _, names in os.walk(out_dir):
        for name in sorted(names):
            full = os.path.join(root, name)
            files[os.path.relpath(full, out_dir).replace(os.sep, '/')] = os.path.getsize(full)
    return files


def _digest(path):
    '''`(sha256[:16], bytes)` of a file.'''
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:16], os.path.getsize(path)


def _parameter_bytes(model):
    '''Unique parameter storage: a tied embedding must be counted once.'''
    seen = {}
    for tensor in model.state_dict().values():
        seen[tensor.data_ptr()] = tensor.numel() * tensor.element_size()
    return sum(seen.values())


def _describe_model(model):
    config = getattr(model, 'config', None)
    if config is None:
        return f'{type(model).__name__}'
    fields = ('model_dim', 'num_heads', 'num_kv_heads', 'head_k_dim', 'block', 'block_n',
              'tied_weight')
    parts = [f'{field}={getattr(config, field)!r}' for field in fields
             if hasattr(config, field)]
    return f'{type(model).__name__}(' + ', '.join(parts) + ')'


def _describe_attention(model):
    from motif.block.attention.chord_attn import ChordAttention
    from motif.block.attention.gdn_attn import GatedDeltaAttention

    chord = delta = None
    for block in model.blocks:
        for layer in block.features:
            if isinstance(layer.attn, ChordAttention) and chord is None:
                chord = layer.attn
            if isinstance(layer.attn, GatedDeltaAttention) and delta is None:
                delta = layer.attn
    parts = []
    if chord is not None:
        parts.append(f'chord heads {chord.num_heads} kv {chord.num_kv_heads} '
                     f'dim {chord.head_dim}')
    if delta is not None:
        parts.append(f'delta dk {delta.dk} dv {delta.dv} conv {delta.conv_size} '
                     f'chunk {delta.chunk_size}')
    return ' | '.join(parts) if parts else 'no attention layers found'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Export a MotifChord model to a decode / prefill ONNX bundle.')
    parser.add_argument('--weights', '--source', dest='source', required=True,
                        help='model checkpoint file or directory holding one')
    parser.add_argument('--precision', choices=PRECISIONS, default='fp16')
    parser.add_argument('--out', dest='out_dir', required=True,
                        help='destination directory for the bundle')
    parser.add_argument('--prefill-lengths', default=','.join(str(length)
                                                              for length in DEFAULT_PREFILL_LENGTHS),
                        help='comma separated static prefill chunk sizes, one graph each')
    parser.add_argument('--prefill-length', dest='prefill_length', type=int, default=None,
                        help='single static prefill chunk size (shorthand for one entry)')
    parser.add_argument('--model-config', default=None,
                        help='explicit config JSON for a checkpoint that carries none')
    parser.add_argument('--no-verify', dest='verify', action='store_false',
                        help='skip the onnxruntime / eager comparison')
    parser.add_argument('--verify-steps', type=int, default=8)
    parser.add_argument('--verify-tokens', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=None,
                        help='onnxruntime intra-op threads used while verifying')
    parser.add_argument('--no-progress', dest='progress', action='store_false')
    parser.add_argument('--keep-solve', dest='solve_patch', action='store_false',
                        help='leave torch.linalg.solve_triangular untouched')
    args = parser.parse_args(argv)
    lengths = (args.prefill_length,) if args.prefill_length else tuple(
        int(part) for part in str(args.prefill_lengths).replace(' ', '').split(',') if part)
    export_onnx(
        args.source, args.precision, args.out_dir,
        prefill_lengths=lengths,
        verify=args.verify,
        verify_steps=args.verify_steps,
        verify_tokens=args.verify_tokens,
        seed=args.seed,
        progress=args.progress,
        threads=args.threads,
        solve_patch=args.solve_patch,
        config=args.model_config,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
