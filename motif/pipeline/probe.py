import inspect
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from motif.model.generate import TextGeneration
from motif.pipeline.callback import Callback


def _encode(tokenizer, text: str, device) -> torch.Tensor:
    '''
    Encode a prompt into a `[1, T]` token tensor.

    Args:
        tokenizer: Tokenizer, optionally exposing `fast_tokenizer`.
        text (str): Prompt text.
        device: Device of the resulting tensor.

    Returns:
        torch.Tensor: `[1, T]` token ids.
    '''
    fast = getattr(tokenizer, 'fast_tokenizer', None)
    if fast is not None:
        ids = fast.encode(text)
        if not isinstance(ids, list):
            ids = ids.ids
    else:
        ids = tokenizer.encode(text)
    return torch.tensor([list(ids)], dtype=torch.long, device=device)


def _decode(tokenizer, ids: List[int]) -> str:
    '''
    Decode token ids, keeping special tokens visible in the probe output.

    Args:
        tokenizer: Tokenizer used for decoding.
        ids (List[int]): Token ids.

    Returns:
        str: The decoded text.
    '''
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(ids)


def _accepted(model) -> Optional[set]:
    '''
    Names of the parameters `model.generate` accepts.

    `_orig_mod` is unwrapped first so that a `torch.compile`d model is handled. None
    is returned when the signature cannot be trusted (no `generate`, an
    introspectable failure, or a `**kwargs` catch-all), which callers treat as "try
    everything".

    Args:
        model: The model to inspect.

    Returns:
        Optional[set]: The accepted parameter names, or None when unknown.
    '''
    target = getattr(model, '_orig_mod', model)
    generate = getattr(target, 'generate', None)
    if generate is None:
        return None
    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def build_generate_kwargs(
    model,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: Optional[int],
) -> Dict[str, Any]:
    '''
    Pick the arguments that `generate` really accepts.

    Args:
        model: Model whose `generate` signature is inspected.
        max_new_tokens (int): Generation budget.
        temperature (float): Sampling temperature, passed directly when the
            signature has it.
        eos_token_id (Optional[int]): End token, when configured.

    Returns:
        Dict[str, Any]: The keyword arguments to pass to `generate`.
    '''
    accepted = _accepted(model)
    kwargs: Dict[str, Any] = {}
    if accepted is None or 'max_new_tokens' in accepted:
        kwargs['max_new_tokens'] = max_new_tokens
    if eos_token_id is not None and (accepted is None or 'eos_token_id' in accepted):
        kwargs['eos_token_id'] = eos_token_id

    if accepted is None or 'temperature' in accepted:
        kwargs['temperature'] = temperature
    elif accepted is not None and 'sampler' in accepted:
        try:
            from motif.model.sampler import Sampler
            kwargs['sampler'] = Sampler(temperature=temperature)
        except Exception:
            pass
    return kwargs


def generate_reply(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    eos_token_id: Optional[int] = None,
) -> str:
    '''
    Let the model continue `prompt` and return the decoded text.

    A mini-motif model (one whose `generate` takes `sampler`) is driven through
    `TextGeneration`, so the prompt is decoded together with the continuation and a
    character split across two tokens is never emitted half-decoded. Codon-style
    models keep the plain `generate` + decode path.

    The model is put in eval mode for the call and restored to its previous mode
    afterwards; failures propagate to the caller, which is expected to catch them.

    Args:
        model: Model to generate with.
        tokenizer: Tokenizer used to encode the prompt and decode the output.
        prompt (str): Prompt text.
        device: Device the prompt is placed on.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        eos_token_id (Optional[int]): End token; None lets a mini-motif model fall
            back to the tokenizer's own eos.

    Returns:
        str: The decoded text, prompt included.
    '''
    was_training = model.training
    model.eval()
    try:
        kwargs = build_generate_kwargs(model, max_new_tokens, temperature, eos_token_id)

        if 'sampler' in kwargs:
            return TextGeneration(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                sampler=kwargs['sampler'],
                device=device,
                skip_special_tokens=False,
                include_prompt=True,
            ).run().text

        ids = _encode(tokenizer, prompt, device)
        with torch.no_grad():
            out = model.generate(ids, **kwargs)
        if hasattr(out, 'run'):
            out = out.run().sequences
        if isinstance(out, (list, tuple)):
            out = torch.tensor(out, dtype=torch.long)
        return _decode(tokenizer, out[0].detach().cpu().tolist())
    finally:
        if was_training:
            model.train()


class Probe(Callback):
    '''
    Run a generation probe every `every_steps` optimiser steps, and once at the
    start of training.

    The probe deliberately adapts to whatever `generate` signature the model exposes
    instead of assuming one interface: a `generate(input_ids, max_new_tokens,
    eos_token_id, sampler, cache)` that returns a `Generation` needing `.run()` and
    carrying **no** `temperature` is driven through
    `motif.model.generate.TextGeneration`, so the probe and the rest of the application
    share one decode path, while a `generate(input_ids, max_new_tokens, sampler,
    temperature, ...)` that returns the token tensor directly is called as it is.
    `build_generate_kwargs()` inspects which parameters `generate` actually accepts
    before deciding what to pass, so neither side needs to change its model code.

    Args:
        prompts (List[str]): Prompt prefixes used by the probe.
        every_steps (int): Interval; 0 runs the probe only at the start of training.
        max_new_tokens (int): Generation budget of every probe.
        temperature (float): Sampling temperature of the probe.
        eos_token_id (Optional[int]): End token, when configured.
        sink: Optional `(tag, text, step)` callback used to forward the generated
            text to other plug-ins (for example `TensorBoardCallback.add_text` or
            `FileLogger.add_text`, so the probe output also shows up on the dashboard
            and in the run log).
    '''

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        every_steps: int = 2000,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
        sink: Optional[Callable[[str, str, int], None]] = None,
    ) -> None:
        '''
        Configure the probe.

        Args:
            prompts (Optional[List[str]]): Prompt prefixes; defaults to `['The']`.
            every_steps (int): Step interval between probes.
            max_new_tokens (int): Generation budget of every probe.
            temperature (float): Sampling temperature of the probe.
            eos_token_id (Optional[int]): End token, when configured.
            sink (Optional[Callable[[str, str, int], None]]): Text sink for the
                generated text.
        '''
        self.prompts = list(prompts) if prompts else ['The']
        self.every_steps = every_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eos_token_id = eos_token_id
        self.sink = sink

    def _run(self, pipeline) -> None:
        '''
        Generate a reply for every prompt and report it.

        A probe failure is written to the console and swallowed: the probe must
        never bring training down. The pipeline is skipped entirely when it has no
        tokenizer.

        Args:
            pipeline: The training pipeline.
        '''
        tokenizer = getattr(pipeline, 'tokenizer', None)
        if tokenizer is None:
            return
        model = getattr(pipeline, 'raw_model', None)
        if model is None:
            model = pipeline.model
        step = pipeline.state.global_step
        for index, prompt in enumerate(self.prompts):
            try:
                text = generate_reply(
                    model, tokenizer, prompt, pipeline.device,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    eos_token_id=self.eos_token_id,
                )
                tqdm.write(f'[probe@{step}] {prompt!r} -> {text!r}')
                if self.sink is not None:
                    self.sink(f'Sanity/probe_{index}', text, step)
            except Exception as exc:
                tqdm.write(f'[!] probe failed for {prompt!r}: {exc}')

    def on_train_start(self, pipeline) -> None:
        '''
        Run the probe once before training starts.

        Args:
            pipeline: The training pipeline.
        '''
        self._run(pipeline)

    def on_step_end(self, pipeline, step_metrics) -> None:
        '''
        Run the probe every `every_steps` steps.

        Args:
            pipeline: The training pipeline.
            step_metrics: Metrics of this step (unused).
        '''
        if self.every_steps and pipeline.state.global_step % self.every_steps == 0:
            self._run(pipeline)
