from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch
from torch import nn

from motif.chat.template import (
    generation_prompt_ids,
    render_prompt,
    split_thought,
    stop_token_ids,
    strip_dialog_tokens,
)
from motif.model.cache import ModelCache
from motif.model.generate import TextGeneration
from motif.model.sampler import Sampler, make_sampler
from motif.tokenizer.tokenizer import PackedTokenizer
from motif.utils.session import resolve_effort


__all__ = ['ROLE_ALIASES', 'ChatMessage', 'ChatReply', 'ChatSession']


#: Role name (and its aliases) -> the role the chat template expects.
ROLE_ALIASES = {
    'system': 'system', 'developer': 'system',
    'user': 'user', 'human': 'user',
    'model': 'model', 'assistant': 'model',
    'tool': 'tool_response', 'tool_response': 'tool_response',
}


@dataclass
class ChatMessage:
    '''
    One turn of the conversation, stored in the shape the chat template wants.

    Attributes:
        role (str): Canonical role ('system' / 'user' / 'model' / 'tool_response').
        content (str): Message text.
        thought (str): Reasoning section of an assistant turn.
        effort (Optional[str]): Effort level of that reasoning section ('low' / 'high'
            / 'max'), so the template renders the same effort marker again.
        tool_calls (List[Any]): Tool calls carried by an assistant turn, passed to the
            template as they are.
        name (Optional[str]): Tool name of a `tool_response` message.
    '''

    role: str
    content: str = ''
    thought: str = ''
    effort: Optional[str] = None
    tool_calls: List[Any] = field(default_factory=list)
    name: Optional[str] = None

    def to_template(self) -> Dict[str, Any]:
        '''
        Returns:
            Dict[str, Any]: This message as a plain dict for `apply_chat_template`.
        '''
        message: Dict[str, Any] = {'role': self.role, 'content': self.content}
        if self.thought:
            message['thought'] = self.thought
        if self.effort is not None:
            message['effort'] = self.effort
        if self.tool_calls:
            message['tool_calls'] = list(self.tool_calls)
        if self.name is not None:
            message['name'] = self.name
        return message


@dataclass
class ChatReply:
    '''
    What one `send()` produced.

    Attributes:
        text (str): The answer, exactly as it is stored in the conversation: the
            conversation markers removed and the whitespace kept, so that the text is
            what the model wrote. It equals `''.join(stream(...))`, and a CLI prints
            `reply.text.strip()`.
        reasoning (str): The reasoning section, markers removed, whitespace kept.
        raw (str): The whole decoded turn, special tokens included.
        streamed (str): What `stream()` yielded, joined - equal to `text` unless the
            caller stopped the stream early, in which case it is the partial answer
            (still with any tool-call markers in it).
        prompt_tokens (int): Tokens actually fed this turn (the whole render when the
            cache could not be reused, only the new tail otherwise).
        new_tokens (int): Tokens sampled this turn.
        steps (int): Decode steps taken.
        reused_cache (bool): Whether the previous turn's cache was continued.
        stopped (bool): Whether a stop token was sampled before the budget ran out.
    '''

    text: str
    reasoning: str
    raw: str
    streamed: str
    prompt_tokens: int
    new_tokens: int
    steps: int
    reused_cache: bool
    stopped: bool


class ChatSession:
    '''
    A conversation with one model: messages in, messages out.

    The session owns the message list, the model cache and the sampler, and renders
    every turn with the tokenizer's chat template. It is single-row by construction
    (`TextGeneration` takes one prompt at a time).

    Attributes:
        model (nn.Module): The model being talked to (put into eval mode).
        tokenizer (PackedTokenizer): Tokenizer carrying the chat template.
        device (torch.device): Device of the prompt tensors.
        sampler (Optional[Sampler]): Sampler bound to every turn (None = greedy).
        stop_ids (Tuple[int, ...]): Tokens that end one turn.
        last_reply (Optional[ChatReply]): Reply of the most recent `send()`.
    '''

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PackedTokenizer,
        system: Optional[str] = None,
        device: Optional[torch.device] = None,
        effort: Optional[str] = None,
        enable_thinking: bool = True,
        enable_think: bool = True,
        enable_effort: bool = True,
        max_new_tokens: int = 256,
        sampler: Optional[Sampler] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None,
        stop: Iterable[Union[str, int]] = (),
        template_kwargs: Optional[Dict[str, Any]] = None,
        reuse_cache: bool = True,
    ) -> None:
        '''
        Start a conversation.

        Args:
            model (nn.Module): Model to generate with (temporarily switched to eval mode).
            tokenizer (PackedTokenizer): Tokenizer exposing `apply_chat_template` /
                `decode`, and carrying the chat template.
            system (Optional[str]): System message to start with.
            device (Optional[torch.device]): Device of the prompt tensor; defaults to
                the model's device.
            effort (Optional[str]): Thinking effort level; None uses the default.
            enable_thinking (bool): Open a reasoning section in the generation prompt.
            enable_think (bool): Keep the reasoning section of the template
                (`<|thought_start|>` / `<|thought_end|>`) in the whole conversation.
                False drops the section from the generation prompt and from every
                model turn, and the answer is read as plain text.
            enable_effort (bool): Emit the effort token inside the reasoning section.
                False leaves `<|thought_start|><|thought_end|>` without a level.
            max_new_tokens (int): Default budget per turn.
            sampler (Optional[Sampler]): Sampler for every turn.
            sampler_kwargs (Optional[Dict[str, Any]]): Arguments for `make_sampler`
                when `sampler` is None; they turn greedy decoding into sampling, so a
                session with neither is greedy.
            stop (Iterable[Union[str, int]]): Extra stop tokens (ids or token names).
            template_kwargs (Optional[Dict[str, Any]]): Extra template switches
                (`effort`, `enable_thinking`, `enable_think` and `enable_effort` are
                filled in automatically).
            reuse_cache (bool): Continue the previous turn's cache (the new ids always
                start with the cached ones when a turn has been generated; the flag only
                switches that off).

        Raises:
            ValueError: If `effort` is not a known level, or if `sampler` and
                `sampler_kwargs` are both given.
        '''
        self.model = model
        self.tokenizer = tokenizer

        parameter = next(model.parameters(), None)
        if device is not None:
            self.device = device
        elif parameter is not None:
            self.device = parameter.device
        else:
            self.device = torch.device('cpu')

        self.effort: str = resolve_effort(effort)
        self.enable_thinking = bool(enable_thinking)
        self.enable_think = bool(enable_think)
        self.enable_effort = bool(enable_effort)
        self.max_new_tokens = int(max_new_tokens)
        self.sampler = make_sampler(sampler, **(sampler_kwargs or {}))
        self.stop_ids = stop_token_ids(tokenizer, extra=stop)
        self.template_kwargs: Dict[str, Any] = dict(template_kwargs or {})
        self.template_kwargs.setdefault('effort', self.effort)
        self.template_kwargs.setdefault('enable_thinking', self.enable_thinking)
        self.template_kwargs.setdefault('enable_think', self.enable_think)
        self.template_kwargs.setdefault('enable_effort', self.enable_effort)
        self.reuse_cache = bool(reuse_cache)

        self._messages: List[ChatMessage] = []
        self._cache = None
        self._prefix_ids: List[int] = []
        self._sequence_ids: List[int] = []
        self._split = 0
        self.last_reply: Optional[ChatReply] = None
        self._was_training = bool(getattr(model, 'training', False))
        self.model.eval()

        if system is not None:
            self.system(system)

    @property
    def messages(self) -> Tuple[ChatMessage, ...]:
        '''
        Returns:
            Tuple[ChatMessage, ...]: The conversation so far, oldest first.
        '''
        return tuple(self._messages)

    @property
    def cache(self) -> Optional[ModelCache]:
        '''
        Returns:
            Optional[ModelCache]: The model cache carried into the next turn (None
            before the first run).
        '''
        return self._cache

    @property
    def history_ids(self) -> List[int]:
        '''
        Returns:
            List[int]: The ids the cache holds, oldest first.
        '''
        return list(self._prefix_ids)

    @property
    def inside_thought(self) -> bool:
        '''
        Whether a generated turn starts inside an open reasoning section.

        It is true only when the prompt actually emitted `<|thought_start|>` plus an
        effort token, which is what `split_thought(..., inside_thought=True)` reads as
        "the text before `<|thought_end|>` is reasoning".

        Returns:
            bool: Whether the generation prompt opened a reasoning section.
        '''
        return bool(
            self.template_kwargs.get('enable_thinking', self.enable_thinking)
            and self.template_kwargs.get('enable_think', self.enable_think)
            and self.template_kwargs.get('enable_effort', self.enable_effort)
        )

    def add(
        self,
        role: str,
        content: str = '',
        *,
        thought: str = '',
        effort: Optional[str] = None,
        tool_calls: Optional[Iterable[Any]] = None,
        name: Optional[str] = None,
    ) -> 'ChatSession':
        '''
        Append one message.

        Args:
            role (str): Role name or alias ('assistant' -> 'model', 'human' -> 'user',
                'tool' -> 'tool_response', 'developer' -> 'system').
            content (str): Message text.
            thought (str): Reasoning section of an assistant turn.
            effort (Optional[str]): Effort level of that reasoning section.
            tool_calls (Optional[Iterable[Any]]): Tool calls of an assistant turn.
            name (Optional[str]): Tool name of a `tool_response` message.

        Returns:
            ChatSession: This session, for chaining.

        Raises:
            ValueError: If the role is unknown.
        '''
        canonical = ROLE_ALIASES.get(str(role).strip().lower())
        if canonical is None:
            raise ValueError(
                f'unknown role {role!r}; expected one of {sorted(ROLE_ALIASES)}'
            )
        self._messages.append(
            ChatMessage(
                role=canonical,
                content='' if content is None else str(content),
                thought=str(thought or ''),
                effort=resolve_effort(effort) if effort is not None else None,
                tool_calls=list(tool_calls or []),
                name=name,
            )
        )
        return self

    def system(self, content: str) -> 'ChatSession':
        '''
        Append a system message.

        Args:
            content (str): System message text.

        Returns:
            ChatSession: This session, for chaining.
        '''
        return self.add('system', content)

    def user(self, content: str) -> 'ChatSession':
        '''
        Append a user message.

        Args:
            content (str): User message text.

        Returns:
            ChatSession: This session, for chaining.
        '''
        return self.add('user', content)

    def model(self, content: str = '', *, thought: str = '', effort: Optional[str] = None) -> 'ChatSession':
        '''
        Append an assistant message.

        Args:
            content (str): Assistant message text.
            thought (str): Reasoning section of that message.
            effort (Optional[str]): Effort level of that reasoning section.

        Returns:
            ChatSession: This session, for chaining.
        '''
        return self.add('model', content, thought=thought, effort=effort)

    def tool(self, content: str, name: Optional[str] = None) -> 'ChatSession':
        '''
        Append a tool response.

        Args:
            content (str): Tool response text.
            name (Optional[str]): Name of the tool that was called.

        Returns:
            ChatSession: This session, for chaining.
        '''
        return self.add('tool_response', content, name=name)

    def reset(self, keep_system: bool = True) -> 'ChatSession':
        '''
        Forget the conversation and the cache.

        Args:
            keep_system (bool): Keep the leading system message.

        Returns:
            ChatSession: This session, for chaining.
        '''
        if keep_system:
            self._messages = [m for m in self._messages if m.role == 'system']
        else:
            self._messages = []
        self._cache = None
        self._prefix_ids = []
        self._sequence_ids = []
        self._split = 0
        self.last_reply = None
        return self

    def render(self, add_generation_prompt: bool = True, **template_kwargs) -> List[int]:
        '''
        Render the conversation so far through the chat template.

        Once a turn has been generated, its ids are kept verbatim and only the
        messages added afterwards go through the template again: the template trims
        the whitespace of a message and drops an empty reasoning section, so
        re-rendering the model's own text would not reproduce the ids the cache
        holds. The result is the exact continuation the next turn feeds.

        Args:
            add_generation_prompt (bool): Append `<|im_start|><|model|>` (and the
                reasoning opener).
            **template_kwargs: Extra template switches for this call only.

        Returns:
            List[int]: Token ids of the rendered conversation.

        Raises:
            ValueError: If the conversation is empty.
        '''
        kwargs = dict(self.template_kwargs)
        kwargs.update(template_kwargs)
        if self._sequence_ids:
            return self._continued_ids(add_generation_prompt, kwargs)
        messages = [message.to_template() for message in self._messages]
        return render_prompt(
            self.tokenizer, messages, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def _continued_ids(self, add_generation_prompt: bool, kwargs: Dict[str, Any]) -> List[int]:
        '''
        Build the ids of the next turn on top of the last generated sequence.

        Args:
            add_generation_prompt (bool): Append the generation prompt.
            kwargs (Dict[str, Any]): Template switches.

        Returns:
            List[int]: The generated ids followed by the render of the new messages.
        '''
        tail = [message.to_template() for message in self._messages[self._split:]]
        if tail:
            rendered = render_prompt(
                self.tokenizer, tail, add_generation_prompt=add_generation_prompt, **kwargs
            )
        elif add_generation_prompt:
            rendered = generation_prompt_ids(
                self.tokenizer,
                effort=kwargs.get('effort', self.effort),
                enable_thinking=kwargs.get('enable_thinking', self.enable_thinking),
                enable_think=kwargs.get('enable_think', self.enable_think),
                enable_effort=kwargs.get('enable_effort', self.enable_effort),
            )
        else:
            rendered = []
        return list(self._sequence_ids) + list(rendered)

    def stream(self, text: Optional[str] = None, **overrides) -> Iterator[str]:
        '''
        Generate one turn, yielding the answer as it appears.

        The reasoning section is not streamed (it is reported as
        `ChatReply.reasoning`), and conversation markers are filtered out, so the
        chunks can be printed as they are. The chunks join into exactly
        `ChatReply.text`, whitespace included. The conversation is committed when the
        stream ends - also when the caller stops early, in which case the partial
        answer is what gets stored.

        Args:
            text (Optional[str]): User message to append first; None continues the
                existing conversation without adding one.
            **overrides: `max_new_tokens`, `eos_token_id` or `sampler` for this turn.

        Yields:
            str: The next piece of the answer ('' is never yielded).

        Raises:
            ValueError: If the conversation is empty.
            TypeError: If an unknown override is given.
        '''
        if text is not None:
            self.user(text)

        ids = self.render()

        base = 0
        cache = None
        reused = False
        if (
            self.reuse_cache
            and self._cache is not None
            and self._prefix_ids
            and len(self._prefix_ids) < len(ids)
            and ids[:len(self._prefix_ids)] == self._prefix_ids
        ):
            base = len(self._prefix_ids)
            cache = self._cache
            reused = True

        max_new_tokens = int(overrides.pop('max_new_tokens', self.max_new_tokens))
        eos_token_id = overrides.pop('eos_token_id', self.stop_ids)
        sampler = overrides.pop('sampler', self.sampler)
        if overrides:
            raise TypeError(f'unexpected keyword arguments: {sorted(overrides)}')

        gen = TextGeneration(
            self.model,
            self.tokenizer,
            torch.tensor([ids], dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            sampler=sampler,
            cache=cache,
            skip_special_tokens=False,
            include_prompt=False,
        )

        emitted = ''
        try:
            for _ in gen:
                split = split_thought(gen.text, inside_thought=self.inside_thought)
                content = strip_dialog_tokens(split.content)
                if content.startswith(emitted):
                    delta = content[len(emitted):]
                else:
                    delta = content
                emitted = content
                if delta:
                    yield delta
        finally:
            self._commit(gen, reused=reused, fed=len(ids) - base, emitted=emitted)

    def send(
        self,
        text: Optional[str] = None,
        *,
        on_chunk: Optional[Callable[[str], None]] = None,
        **overrides,
    ) -> ChatReply:
        '''
        Generate one turn to completion.

        Args:
            text (Optional[str]): User message to append first; None continues the
                existing conversation without adding one.
            on_chunk (Optional[Callable[[str], None]]): Callback invoked with every
                streamed chunk.
            **overrides: `max_new_tokens`, `eos_token_id` or `sampler` for this turn.

        Returns:
            ChatReply: The reply, already appended to the conversation.

        Raises:
            ValueError: If the conversation is empty.
            TypeError: If an unknown override is given.
        '''
        for chunk in self.stream(text, **overrides):
            if on_chunk is not None:
                on_chunk(chunk)
        if self.last_reply is None:
            raise RuntimeError('the reply was not committed')
        return self.last_reply

    def _commit(self, gen: TextGeneration, reused: bool, fed: int, emitted: str) -> None:
        '''
        Store the finished turn: the assistant message, the ids now in the cache and
        the `ChatReply`.

        Args:
            gen (TextGeneration): The finished run.
            reused (bool): Whether it continued the previous cache.
            fed (int): Number of prompt tokens fed this turn.
            emitted (str): The answer streamed so far (used as `streamed`).

        Returns:
            None
        '''
        raw = gen.text
        split = split_thought(raw, inside_thought=self.inside_thought)
        content = strip_dialog_tokens(split.content)
        reasoning = strip_dialog_tokens(split.reasoning)
        text = content

        self._messages.append(
            ChatMessage(role='model', content=content, thought=reasoning, effort=split.effort)
        )

        sequence = [int(token) for token in gen.sequences[0].tolist()]
        covered = int(gen.cache.seq_len) if gen.cache is not None else 0
        covered = max(0, min(covered, len(sequence)))
        self._prefix_ids = sequence[:covered]
        self._sequence_ids = sequence
        self._split = len(self._messages)
        self._cache = gen.cache

        generated = [int(token) for token in gen.generated[0].tolist()]
        stopped = bool(gen.eos_ids) and any(token in gen.eos_ids for token in generated)

        self.last_reply = ChatReply(
            text=text,
            reasoning=reasoning,
            raw=raw,
            streamed=content if emitted == '' else emitted,
            prompt_tokens=int(fed),
            new_tokens=len(generated),
            steps=int(gen.steps),
            reused_cache=bool(reused),
            stopped=bool(stopped),
        )

    def close(self) -> None:
        '''
        Drop the cache and put the model back into the mode it was in.

        Returns:
            None
        '''
        self._cache = None
        self._prefix_ids = []
        self._sequence_ids = []
        self._split = 0
        self.model.train(self._was_training)

    def __enter__(self) -> 'ChatSession':
        '''
        Returns:
            ChatSession: This session.
        '''
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        '''
        Args:
            exc_type (Any): Exception type, when the block raised.
            exc (Any): Exception instance, when the block raised.
            traceback (Any): Traceback, when the block raised.

        Returns:
            None
        '''
        self.close()

    def __repr__(self) -> str:
        '''
        Returns:
            str: Message count, cached positions and the model's mode.
        '''
        return (
            f'ChatSession(messages={len(self._messages)}, '
            f'cached={len(self._prefix_ids)}, training={self.model.training})'
        )
