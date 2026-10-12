import os
import time

import numpy as np
import onnxruntime as ort

from motif.onnx.runtime.tokenizer import PackedTokenizer

DTYPES = {
    'tensor(float16)': np.float16,
    'tensor(float)': np.float32,
    'tensor(float64)': np.float64,
    'tensor(int64)': np.int64,
    'tensor(int32)': np.int32,
    'tensor(bool)': np.bool_,
}


def graph_path(bundle, name):
    for candidate in (os.path.join(bundle, f'{name}.onnx'),
                      os.path.join(bundle, name, f'{name}.onnx')):
        if os.path.exists(candidate):
            return candidate
    return None


def prefill_graph_paths(bundle):
    '''
    `{length: path}` for every prefill graph under `bundle`.

    The exporter writes `prefill_<T>.onnx` next to the decode graph, or
    `prefill_<T>/prefill_<T>.onnx` when the graphs ended up with separate
    weight blobs.
    '''
    found = {}
    for name in sorted(os.listdir(bundle)):
        path = None
        if name.startswith('prefill_') and name.endswith('.onnx'):
            path = os.path.join(bundle, name)
        elif name.startswith('prefill_') and os.path.isdir(os.path.join(bundle, name)):
            candidate = os.path.join(bundle, name, f'{name}.onnx')
            if os.path.exists(candidate):
                path = candidate
        if path is None:
            continue
        length = name[len('prefill_'):].split('.')[0]
        if length.isdigit():
            found[int(length)] = path
    return found


class OrtEngine:
    '''A MotifChord model driven entirely by onnxruntime.'''

    def __init__(self, bundle, tokenizer_path, providers=None, verbose=False):
        self.bundle = bundle
        self.verbose = verbose
        self.tokenizer = PackedTokenizer(tokenizer_path)
        self.providers = list(providers or ['CPUExecutionProvider'])
        ort.set_default_logger_severity(3)
        self.session_decode = self._session('decode')
        if self.session_decode is None:
            raise FileNotFoundError(f'no decode graph under {bundle}')
        self.inputs = self.session_decode.get_inputs()
        self.names = [i.name for i in self.inputs]
        self.prefill_sessions = self._prefill_sessions()
        self.prefill_used = bool(self.prefill_sessions)
        if verbose:
            self.describe()

    def _session(self, name):
        path = graph_path(self.bundle, name)
        if path is None:
            return None
        return ort.InferenceSession(path, providers=self.providers)

    def _prefill_sessions(self):
        '''Every usable prefill chunk, keyed by the chunk size baked into its graph.'''
        return {length: ort.InferenceSession(path, providers=self.providers)
                for length, path in prefill_graph_paths(self.bundle).items()}

    @property
    def prefill_lengths(self):
        '''Available chunk sizes, smallest first.'''
        return sorted(self.prefill_sessions)

    @property
    def session_prefill(self):
        '''The largest prefill graph, for callers that only need one.'''
        return self.prefill_sessions[max(self.prefill_sessions)] if self.prefill_sessions else None

    def describe(self):
        for label, session in (('decode', self.session_decode), ('prefill', self.session_prefill)):
            if session is None:
                print(f'[info] {label}: absent', flush=True)
                continue
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            print(f'[info] {label}: {len(inputs)} inputs, {len(outputs)} outputs, '
                  f'input_ids {inputs[0].shape}, position {inputs[1].type}, '
                  f'logits {outputs[0].shape} {outputs[0].type}', flush=True)
        if self.prefill_lengths:
            print(f'[info] prefill chunks {self.prefill_lengths}', flush=True)

    def initial_state(self):
        '''Zero cache tensors, shaped and typed after the graphs' own inputs.'''
        state = []
        for spec in self.inputs[2:]:
            shape = [dim if isinstance(dim, int) else 0 for dim in spec.shape]
            state.append(np.zeros(shape, dtype=DTYPES.get(spec.type, np.float32)))
        return state

    def logits_dtype(self):
        return DTYPES.get(self.session_decode.get_outputs()[0].type, np.float32)

    def _run(self, ids, position, state, session):
        feeds = {
            'input_ids': np.asarray(ids, dtype=np.int64).reshape(1, -1),
            'position': np.array([position], dtype=np.int64),
        }
        feeds.update(dict(zip(self.names[2:], state)))
        outputs = session.run(None, feeds)
        return outputs[0], list(outputs[1:])

    def step(self, token, position, state):
        '''One token through the decode graph.'''
        logits, state = self._run([token], position, state, self.session_decode)
        return logits[0, 0], state

    def prefill(self, ids, state):
        '''
        Consume a prompt: the largest chunk that fits, the tail one token at a
        time through the decode graph.
        '''
        ids = list(ids)
        position = 0
        logits = None
        while position < len(ids):
            remaining = len(ids) - position
            chunk = self.chunk_for(remaining)
            if chunk is None:
                logits, state = self.step(ids[position], position, state)
                position += 1
                continue
            try:
                block, state = self._run(ids[position:position + chunk], position, state,
                                         self.prefill_sessions[chunk])
            except Exception as exc:
                print(f'[warn] the prefill graph for {chunk} tokens failed '
                      f'({type(exc).__name__}: {exc}); falling back to the decode graph',
                      flush=True)
                self.prefill_sessions.pop(chunk, None)
                self.prefill_used = bool(self.prefill_sessions)
                continue
            logits = block[0, -1]
            position += chunk
        return logits, state, position

    def chunk_for(self, remaining):
        '''The largest available chunk that still fits in `remaining` tokens.'''
        for length in reversed(self.prefill_lengths):
            if remaining >= length:
                return length
        return None

    def decode(self, ids):
        return self.tokenizer.decode(list(ids))

    def generate(self, prompt_ids, sampler, max_new_tokens=64, stop_ids=(), on_token=None):
        '''Prefill, then decode until a stop token or `max_new_tokens`.'''
        prompt_ids = list(prompt_ids)
        started = time.perf_counter()
        state = self.initial_state()
        logits, state, position = self.prefill(prompt_ids, state)
        prefill_seconds = time.perf_counter() - started
        produced = []
        reason = 'length'
        while len(produced) < max_new_tokens:
            token = sampler(logits, seen=prompt_ids + produced)
            if token in stop_ids:
                reason = 'stop'
                break
            produced.append(token)
            if on_token is not None:
                on_token(produced)
            logits, state = self.step(token, position, state)
            position += 1
        elapsed = time.perf_counter() - started
        return {
            'ids': produced,
            'text': self.decode(produced),
            'reason': reason,
            'prompt_tokens': len(prompt_ids),
            'prefill_seconds': prefill_seconds,
            'seconds': elapsed,
            'tokens_per_second': len(produced) / elapsed if elapsed > 0 else 0.0,
            'position': position,
            'prefill_graph': self.prefill_used and bool(self.prefill_sessions),
        }
