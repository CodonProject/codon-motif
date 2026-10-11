from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch

from motif.model.cache import ModelCache
from motif.model.sampler import Sampler

if TYPE_CHECKING:
    from motif.model.motif import MotifChordModel
    from motif.tokenizer.tokenizer import PackedTokenizer


EosTokenId = Union[int, str, Sequence[Union[int, str]], None]


class Generation:
    '''
    Generator plus result snapshot.

    Iterating it runs the generation loop; once the iteration ends (or part-way
    through) `sequences` / `generated` / `cache` / `steps` / `finished` can be read
    directly. Nothing runs until it is iterated, and `cache` keeps whatever was
    passed in.

        gen = model.generate(prompt, max_new_tokens=64)
        for token in gen:
            print(int(token))
        nxt = model.generate(gen.sequences, cache=gen.cache)

    Each step is: forward (with the cache) -> `next_token` (given the full sequence,
    which is what makes the repetition penalty effective) -> append the token. The
    first forward feeds the whole input (prefill); afterwards only the token of the
    previous step is fed.

    When a cache is passed, the first `cache.seq_len` positions of `input_ids` count
    as already cached and are cut off, so a resumed run can pass the full sequence
    as it is.

    A `mask` (keep-mask over the key axis, `True` on positions that may be attended)
    covers the positions fed on this call only, exactly like `input_ids`: when a
    cache is passed its first `cache.seq_len` rows are cut off as already-cached, and
    it grows by one all-`True` column per step so that it stays aligned with the
    appended tokens.

    Once a row of the batch samples eos it is fed an eos placeholder to keep the
    positions aligned; the iteration stops when every row has finished or
    `max_new_tokens` is reached.

    Attributes:
        sequences (torch.Tensor): `[B, prompt_len + steps]` full sequence, prompt
            and the prefix already stored in the cache included.
        cache (Optional[ModelCache]): Current cache; after the iteration it is the
            final one and generation can continue from it directly.
        steps (int): Number of steps taken so far.
        finished (torch.Tensor): `[B]` bool flags telling whether a row stopped
            because it sampled eos.
    '''

    def __init__(
        self,
        model: 'MotifChordModel',
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: EosTokenId = None,
        sampler: Optional[Sampler] = None,
        cache: Optional[ModelCache] = None,
        mask: Optional[torch.Tensor] = None,
        tokenizer=None,
    ) -> None:
        '''
        Validate the inputs and seed the generation state.

        Args:
            model ('MotifChordModel'): Model called on every step.
            input_ids (torch.Tensor): `[B, T]` prompt, including any prefix that is
                already in `cache`.
            max_new_tokens (int): Maximum number of tokens to generate.
            eos_token_id (EosTokenId): Token id, token name the `tokenizer` resolves,
                sequence of either, or None to disable early stopping.
            sampler (Optional[Sampler]): Sampler used for this run.
            cache (Optional[ModelCache]): Cache to continue from.
            mask (Optional[torch.Tensor]): Keep-mask over the key axis, handed to the
                model on every step. `[B, T]` is the usual shape; it is sliced by the
                already-cached prefix in the same way as `input_ids`.
            tokenizer: Tokenizer able to resolve token names, needed only when an eos
                is spelled as a name.

        Raises:
            ValueError: If `input_ids` is not 2-dimensional, if `max_new_tokens` is
                negative, or if the cache already covers the whole input so that no
                new token is left to feed.
        '''
        if input_ids.dim() != 2:
            raise ValueError(f'input_ids must be [B, T], got {tuple(input_ids.shape)}')
        if int(max_new_tokens) < 0:
            raise ValueError(f'max_new_tokens must not be negative, got {max_new_tokens}')

        cached = cache.seq_len if cache is not None else 0
        if cached >= input_ids.shape[1]:
            raise ValueError(
                f'the cache already holds {cached} positions but the input has only '
                f'{input_ids.shape[1]} tokens, so there is nothing new to feed'
            )

        self.model = model
        self.cache = cache
        self.sampler = sampler
        self.max_new_tokens = int(max_new_tokens)
        self.sequences = input_ids
        self.prompt_len = input_ids.shape[1]
        self._next_input = input_ids[:, cached:]
        self._key_mask = None if mask is None else mask[:, :cached + input_ids.shape[1]]

        self.eos_ids = _resolve_eos(eos_token_id, tokenizer)
        self.fill_id = min(self.eos_ids) if self.eos_ids else None

        self.finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.steps = 0

    @property
    def done(self) -> bool:
        '''
        Returns:
            bool: True once `max_new_tokens` steps were taken or every row of the
            batch sampled eos.
        '''
        return self.steps >= self.max_new_tokens or bool(self.finished.all())

    @property
    def generated(self) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor: `[B, steps]` tokens produced by this run, prompt and the
            prefix already in the cache excluded.
        '''
        return self.sequences[:, self.prompt_len:]

    def step(self) -> torch.Tensor:
        '''
        Run a single decode step.

        Returns:
            torch.Tensor: `[B, 1]` token sampled for this step; rows that already
            finished get the eos placeholder.
        '''
        with torch.no_grad():
            out = self.model(
                self._next_input,
                cache=self.cache,
                mask=self._key_mask,
            )
            token = out.next_token(sampler=self.sampler, input_ids=self.sequences)

        self.cache = out.cache

        if self.fill_id is not None:
            token = token.masked_fill(self.finished.unsqueeze(1), self.fill_id)

        self.sequences = torch.cat([self.sequences, token], dim=1)
        if self._key_mask is not None:
            keep = torch.ones_like(token, dtype=torch.bool)
            self._key_mask = torch.cat([self._key_mask, keep], dim=1)
        self._next_input = token
        self.steps += 1

        if self.eos_ids:
            hit = torch.zeros_like(self.finished)
            for eos_id in self.eos_ids:
                hit |= token.squeeze(1) == eos_id
            self.finished |= hit
        return token

    def __iter__(self) -> 'Generation':
        '''Returns: Generation: This generator.'''
        return self

    def __next__(self) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor: The token of the next step.

        Raises:
            StopIteration: Once the generator is done.
        '''
        if self.done:
            raise StopIteration
        return self.step()

    def run(self) -> 'Generation':
        '''
        Run to completion, for callers that do not care about the intermediate
        tokens.

        Returns:
            Generation: This generator, now holding the final result.
        '''
        for _ in self:
            pass
        return self

    def __repr__(self) -> str:
        '''
        Returns:
            str: A short debug representation with the prompt length, the number of
            steps and how many rows have finished.
        '''
        batch = self.sequences.shape[0]
        done = int(self.finished.sum())
        return (
            f'Generation(prompt_len={self.prompt_len}, steps={self.steps}, '
            f'generated={self.generated.shape[1]}, finished={done}/{batch}, '
            f'done={self.done})'
        )


_REPLACEMENT_CHAR = '\ufffd'


def _complete_text(text: str) -> str:
    '''
    Hold back a trailing half-decoded byte-level character.

    The tokenizer runs on bytes, so a character whose UTF-8 sequence is split across
    two tokens decodes to U+FFFD until the next token completes it. That happens on
    most multi-byte characters (every Chinese character is three bytes), so a naive
    per-token decode produces mojibake in the middle of the stream.

    Args:
        text (str): Freshly decoded text.

    Returns:
        str: `text` without its trailing replacement characters.
    '''
    return text.rstrip(_REPLACEMENT_CHAR)


def _tokenizer_eos_id(tokenizer) -> Optional[int]:
    '''
    Read the end-of-sequence id off a tokenizer, when it advertises one.

    Args:
        tokenizer: Tokenizer, optionally exposing `token_eos` and `token_to_id`.

    Returns:
        Optional[int]: The eos token id, or None when it cannot be determined.
    '''
    name = getattr(tokenizer, 'token_eos', None)
    to_id = getattr(tokenizer, 'token_to_id', None)
    if not isinstance(name, str) or to_id is None:
        return None
    try:
        token_id = to_id(name)
    except Exception:
        return None
    return None if token_id is None else int(token_id)


def _decode_ids(tokenizer, ids, skip_special_tokens: bool) -> str:
    '''
    Decode token ids, tolerating a tokenizer without `skip_special_tokens`.

    Args:
        tokenizer: Tokenizer used for decoding.
        ids: Sequence of token ids.
        skip_special_tokens (bool): Drop special tokens from the output.

    Returns:
        str: The decoded text.
    '''
    try:
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        return tokenizer.decode(ids)


def _as_sequence(value) -> list:
    '''
    Wrap a single token specification into a list.

    Args:
        value: A token id, a token name, None, or an already iterable of those.

    Returns:
        list: The values to resolve; an empty list for None.
    '''
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _resolve_eos(eos_token_id, tokenizer=None) -> set:
    '''
    Turn the eos spellings into one set of token ids.

    A token may be given as an id or as a name. Names need a tokenizer, because
    `tokenizer.token_eos` is a name such as `'<|im_end|>'` and not an id.

    Args:
        eos_token_id: Token id, token name, sequence of those, or None.
        tokenizer: Tokenizer that resolves the names.

    Returns:
        set: The eos token ids.

    Raises:
        ValueError: If a name was given without a tokenizer, or one the tokenizer
            does not know.
    '''
    resolved = set()
    for eos in _as_sequence(eos_token_id):
        if isinstance(eos, str):
            to_id = getattr(tokenizer, 'token_to_id', None)
            if to_id is None:
                raise ValueError(
                    f'eos token {eos!r} is a name, so the generation needs a tokenizer '
                    f'able to resolve it; pass its id instead'
                )
            token_id = to_id(eos)
            if token_id is None:
                raise ValueError(f'the tokenizer does not know the eos token {eos!r}')
            eos = token_id
        resolved.add(int(eos))
    return resolved


def _model_device(model) -> torch.device:
    '''
    Read the device a model's parameters live on.

    Args:
        model: Module whose parameters decide the device.

    Returns:
        torch.device: Device of the first parameter, CPU for a parameterless model.
    '''
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device('cpu')


def _encode_prompt(tokenizer, text: str, device, apply_safe_rule: bool) -> torch.Tensor:
    '''
    Encode a prompt string into a `[1, T]` token tensor.

    Args:
        tokenizer: Tokenizer exposing `encode`.
        text (str): Prompt text.
        device: Device of the resulting tensor.
        apply_safe_rule (bool): Escape structural characters first, when supported.

    Returns:
        torch.Tensor: `[1, T]` token ids on `device`.
    '''
    try:
        ids = tokenizer.encode(text, apply_safe_rule=apply_safe_rule)
    except TypeError:
        ids = tokenizer.encode(text)
    return torch.tensor([list(ids)], dtype=torch.long, device=device)


class TextGeneration(Generation):
    '''
    Text in, text out: `Generation` plus a tokenizer.

        gen = TextGeneration(model, tokenizer, '从前有座山，')
        for chunk in gen:                     # text produced by this step
            print(chunk, end='', flush=True)
        print(gen.text)

        TextGeneration(model, tokenizer, prompt, eos_token_id=()).run().text

    Only a single-row prompt is supported (`str` or `[1, T]`); generate a batch of
    prompts one at a time (plain `Generation` still takes any batch size).

    When `eos_token_id` is None the tokenizer's own end token is used
    (`token_eos` -> `token_to_id`), so a chat tokenizer stops at its `<|im_end|>`
    without being told; pass `eos_token_id=()` to switch early stopping off.

    Attributes:
        tokenizer: Tokenizer used to encode the prompt and decode the text.
        prompt_text (str): The decoded prompt.
        text (str): Text produced so far - the continuation, or prompt + continuation
            when `include_prompt` is set. Empty until the generation runs.
        include_prompt (bool): Whether `text` starts with the prompt.
        skip_special_tokens (bool): Passed to `decode`.
    '''

    def __init__(
        self,
        model: 'MotifChordModel',
        tokenizer,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 32,
        eos_token_id: EosTokenId = None,
        sampler: Optional[Sampler] = None,
        cache: Optional[ModelCache] = None,
        device=None,
        skip_special_tokens: bool = True,
        include_prompt: bool = False,
        apply_safe_rule: bool = True,
    ) -> None:
        '''
        Validate the prompt and seed the generation state.

        Args:
            model ('MotifChordModel'): Model called on every step.
            tokenizer: Tokenizer exposing `encode` / `decode` (`token_eos` and
                `token_to_id` give the default eos).
            prompt (Union[str, torch.Tensor]): Prompt text, or a `[1, T]` id tensor
                (used as it is, so `device` is ignored).
            max_new_tokens (int): Maximum number of tokens to generate.
            eos_token_id (EosTokenId): Token id, sequence of ids, or None to take the
                tokenizer's eos; `()` disables early stopping.
            sampler (Optional[Sampler]): Sampler used for this run.
            cache (Optional[ModelCache]): Cache to continue from.
            device: Device of the encoded prompt when `prompt` is text; None uses the
                device of the model's parameters. A tensor prompt is used as it is and
                has to be on that device.
            skip_special_tokens (bool): Drop special tokens while decoding.
            include_prompt (bool): Let `text` start with the prompt.
            apply_safe_rule (bool): Escape structural characters while encoding text
                (ignored when `prompt` is already token ids).

        Raises:
            ValueError: If `prompt` is a tensor that is not `[1, T]`, or one that lives
                on another device than the model's parameters.
        '''
        if isinstance(prompt, str):
            if device is None:
                device = _model_device(model)
            ids = _encode_prompt(tokenizer, prompt, device, apply_safe_rule)
        else:
            ids = prompt
            if ids.dim() != 2 or ids.shape[0] != 1:
                raise ValueError(
                    f'prompt must be a string or [1, T] for text generation, '
                    f'got {tuple(ids.shape)}'
                )
            model_device = _model_device(model)
            if ids.device != model_device:
                raise ValueError(
                    f'prompt ids are on {ids.device} but the model is on {model_device}; '
                    f'move the tensor with .to({model_device}) or pass the prompt as text'
                )

        if eos_token_id is None:
            eos_token_id = _tokenizer_eos_id(tokenizer)

        self.tokenizer = tokenizer
        self.skip_special_tokens = bool(skip_special_tokens)
        self.include_prompt = bool(include_prompt)
        self.prompt_text = _decode_ids(tokenizer, ids[0].tolist(), skip_special_tokens)

        super().__init__(model, ids, max_new_tokens, eos_token_id, sampler, cache,
                         tokenizer=tokenizer)

        self._emitted = self._safe_text() if self.include_prompt else ''

    @property
    def text(self) -> str:
        '''
        Returns:
            str: The text produced so far - continuation only, or prompt +
            continuation when `include_prompt` is set. A character whose bytes were
            not complete when generation stopped is not included.
        '''
        return self._emitted

    def _safe_text(self) -> str:
        '''
        Returns:
            str: The whole decoded sequence (`sequences` with `include_prompt`, the
            generated part alone otherwise), with a trailing half-decoded character
            held back.
        '''
        ids = self.sequences[0] if self.include_prompt else self.generated[0]
        return _complete_text(_decode_ids(self.tokenizer, ids.tolist(), self.skip_special_tokens))

    def step(self) -> str:
        '''
        Run a single decode step.

        Returns:
            str: The text this step produced. It is '' while the only new bytes still
            leave a character incomplete, and after every row has finished.
        '''
        super().step()
        full = self._safe_text()
        if full.startswith(self._emitted):
            delta = full[len(self._emitted):]
        else:
            delta = full
        self._emitted = full
        return delta
