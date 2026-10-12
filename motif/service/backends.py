import dataclasses
import os
import time
from typing import Dict, Iterator, List, Optional, Sequence

from motif.onnx.runtime import paths as runtime_paths
from motif.onnx.runtime import template as runtime_template

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_CONTEXT_LENGTH = 4096


@dataclasses.dataclass
class SamplingParams:
    '''One request's decoding knobs, already resolved against the defaults.'''

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    seed: Optional[int] = None
    stop_ids: Sequence[int] = ()


class OnnxBackend:
    '''An exported ONNX bundle, driven by `onnxruntime` (CPU by default).'''

    name = 'onnx'
    weights = 'onnx runtime'

    def __init__(self, bundle=None, tokenizer=None, providers=None, model_id=None,
                 context_length=DEFAULT_CONTEXT_LENGTH, system=None, verbose=False):
        from motif.onnx.runtime.engine import OrtEngine
        chosen = bundle or runtime_paths.default_bundle()
        if chosen is None:
            raise FileNotFoundError(
                'no ONNX bundle given: pass the directory holding decode.onnx, '
                'prefill_<T>.onnx and weights.bin, or set '
                f'{runtime_paths.BUNDLE_ENV}')
        self.bundle = os.path.abspath(chosen)
        self.tokenizer_path = os.path.abspath(tokenizer or runtime_paths.default_tokenizer())
        if not os.path.isdir(self.bundle):
            raise FileNotFoundError(
                f'no ONNX bundle at {self.bundle}: export one with '
                f'`python -m motif.onnx.export_onnx`')
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f'no tokenizer archive at {self.tokenizer_path}')
        self.engine = OrtEngine(self.bundle, self.tokenizer_path, providers=providers,
                                verbose=verbose)
        self.tokenizer = self.engine.tokenizer
        self.template = runtime_template
        self.model_id = model_id or 'codon-motif-onnx'
        self.context_length = int(context_length)
        self.system = system
        self.device = 'cpu'

    def _sampler(self, params: SamplingParams):
        from motif.onnx.runtime.sampler import NumpySampler
        return NumpySampler(temperature=params.temperature, top_k=params.top_k,
                            top_p=params.top_p,
                            repetition_penalty=params.repetition_penalty, seed=params.seed)

    def info(self) -> Dict[str, object]:
        return {
            'backend': self.name,
            'weights': self.weights,
            'model_id': self.model_id,
            'device': self.device,
            'bundle': self.bundle,
            'tokenizer': self.tokenizer_path,
            'providers': list(self.engine.providers),
            'prefill_chunks': list(self.engine.prefill_lengths),
            'context_length': self.context_length,
            'vocab_size': self.tokenizer.vocab_size,
        }

    def render(self, messages, thinking=True, effort=None, tools=None) -> List[int]:
        kwargs = {
            'add_generation_prompt': True,
            'enable_thinking': bool(thinking),
            'effort': effort or self.template.EFFORT_DEFAULT,
        }
        if tools:
            kwargs['tools'] = tools
        return self.template.render_prompt(self.tokenizer, messages, **kwargs)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def stop_ids(self, extra=()) -> Sequence[int]:
        return self.template.stop_token_ids(self.tokenizer, extra)

    def stop_string_ids(self, text: str) -> List[int]:
        '''The ids a stop string encodes to; the runtime stops on tokens.'''
        return list(self.tokenizer.encode(text, apply_safe_rule=False))

    def generate(self, prompt_ids, params: SamplingParams) -> Dict[str, object]:
        result = self.engine.generate(list(prompt_ids), self._sampler(params),
                                      max_new_tokens=params.max_new_tokens,
                                      stop_ids=self.stop_ids(params.stop_ids))
        return {
            'ids': list(result['ids']),
            'text': result['text'],
            'reason': result['reason'],
            'prompt_tokens': result['prompt_tokens'],
            'completion_tokens': len(result['ids']),
            'seconds': result['seconds'],
            'tokens_per_second': result['tokens_per_second'],
        }

    def stream(self, prompt_ids, params: SamplingParams) -> Iterator[str]:
        '''
        Text deltas, one per decoded step. `engine.generate` returns the whole
        reply at once, so the step loop is repeated here with sampling on.
        '''
        sampler = self._sampler(params)
        stop = set(self.stop_ids(params.stop_ids))
        prompt_ids = list(prompt_ids)
        state = self.engine.initial_state()
        logits, state, position = self.engine.prefill(prompt_ids, state)
        seen = list(prompt_ids)
        produced: List[int] = []
        text = ''
        while len(produced) < params.max_new_tokens:
            token = int(sampler(logits, seen=seen))
            if token in stop:
                return
            produced.append(token)
            seen.append(token)
            full = self.engine.decode(produced)
            delta = full[len(text):] if full.startswith(text) else full
            text = full
            if delta:
                yield delta
            logits, state = self.engine.step(token, position, state)
            position += 1


class TorchBackend:
    '''A `MotifChordModel`, driven by `motif` and `torch`.'''

    name = 'torch'
    weights = 'torch'

    def __init__(self, model, tokenizer=None, tokenizer_path=None, model_id=None,
                 context_length=DEFAULT_CONTEXT_LENGTH, system=None, device=None,
                 verbose=False):
        import torch
        from motif.model.generate import TextGeneration
        from motif.model.sampler import Sampler
        from motif.tokenizer import PackedTokenizer
        self.torch = torch
        self._text_generation = TextGeneration
        self._sampler_cls = Sampler
        self.model = model
        self.model.eval()
        self.device = torch.device(device) if device else self._model_device()
        path = tokenizer_path or runtime_paths.default_tokenizer()
        self.tokenizer = tokenizer or PackedTokenizer(path)
        self.tokenizer_path = os.path.abspath(path)
        self.template = runtime_template
        self.model_id = model_id or 'codon-motif-torch'
        self.context_length = int(context_length)
        self.system = system
        self.verbose = verbose

    def _model_device(self):
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else self.torch.device('cpu')

    def info(self) -> Dict[str, object]:
        return {
            'backend': self.name,
            'weights': self.weights,
            'model_id': self.model_id,
            'device': str(self.device),
            'model_class': type(self.model).__name__,
            'tokenizer': self.tokenizer_path,
            'context_length': self.context_length,
            'vocab_size': self.tokenizer.vocab_size,
        }

    def render(self, messages, thinking=True, effort=None, tools=None) -> List[int]:
        kwargs = {
            'add_generation_prompt': True,
            'enable_thinking': bool(thinking),
            'effort': effort or self.template.EFFORT_DEFAULT,
        }
        if tools:
            kwargs['tools'] = tools
        return self.template.render_prompt(self.tokenizer, messages, **kwargs)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def stop_ids(self, extra=()) -> Sequence[int]:
        return self.template.stop_token_ids(self.tokenizer, extra)

    def stop_string_ids(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, apply_safe_rule=False))

    def _sampler(self, params: SamplingParams):
        return self._sampler_cls(temperature=params.temperature, top_k=params.top_k,
                                 top_p=params.top_p,
                                 repetition_penalty=params.repetition_penalty)

    def _generation(self, prompt_ids, params: SamplingParams):
        '''A `TextGeneration` that keeps the protocol markers in its text.'''
        ids = self.torch.tensor([list(prompt_ids)], dtype=self.torch.long,
                                device=self.device)
        return self._text_generation(
            self.model, self.tokenizer, ids,
            max_new_tokens=params.max_new_tokens,
            eos_token_id=list(self.stop_ids(params.stop_ids)),
            sampler=self._sampler(params),
            cache=self.model.create_cache(),
            skip_special_tokens=False)

    def stream(self, prompt_ids, params: SamplingParams) -> Iterator[str]:
        generation = self._generation(prompt_ids, params)
        with self.torch.no_grad():
            for delta in generation:
                if delta:
                    yield delta

    def generate(self, prompt_ids, params: SamplingParams) -> Dict[str, object]:
        generation = self._generation(prompt_ids, params)
        started = time.perf_counter()
        parts: List[str] = []
        with self.torch.no_grad():
            for delta in generation:
                if delta:
                    parts.append(delta)
        elapsed = time.perf_counter() - started
        ids = generation.generated[0].tolist() if generation.steps else []
        text = ''.join(parts)
        return {
            'ids': ids,
            'text': text,
            'reason': 'stop' if bool(generation.finished.any()) else 'length',
            'prompt_tokens': len(list(prompt_ids)),
            'completion_tokens': len(ids),
            'seconds': elapsed,
            'tokens_per_second': len(ids) / elapsed if elapsed > 0 else 0.0,
        }


def build_backend(model=None, model_path=None, bundle=None, tokenizer=None, providers=None,
                  model_id=None, context_length=DEFAULT_CONTEXT_LENGTH, system=None,
                  device=None, verbose=False):
    '''
    Build the backend a request should use.

    A `model` instance or a `model_path` selects the torch backend, which is the
    only path that imports `motif`; anything else selects the ONNX backend.
    '''
    if model is not None:
        return TorchBackend(model, tokenizer=tokenizer, tokenizer_path=tokenizer,
                            model_id=model_id, context_length=context_length,
                            system=system, device=device, verbose=verbose)
    if model_path is not None:
        from motif.model import MotifChordModel
        loaded = MotifChordModel.from_pretrained(model_path, device=device, verbose=verbose)
        return TorchBackend(loaded, tokenizer=tokenizer, tokenizer_path=tokenizer,
                            model_id=model_id, context_length=context_length,
                            system=system, device=device, verbose=verbose)
    return OnnxBackend(bundle=bundle, tokenizer=tokenizer, providers=providers,
                       model_id=model_id, context_length=context_length, system=system,
                       verbose=verbose)
