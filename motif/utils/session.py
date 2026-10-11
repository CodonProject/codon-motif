import copy
import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional, Sequence, Union

import torch

from motif.tokenizer.tokenizer import PackedTokenizer

#: How the supervised mask is built for a rendered conversation.
MaskPolicy = Union[Literal['all', 'none', 'content', 'thought', 'answer', 'fim'], Sequence[bool]]

#: Logical name -> vocabulary token text. The modality names are absent on purpose:
#: their vocabulary slots stay reserved (`motif/tokenizer/spec_token.py`) but nothing in
#: this library writes or reads them.
TOKEN_NAMES = {
    'bos': '<|bos|>', 'eos': '<|eos|>', 'end_of_text': '<|end_of_text|>',
    'im_start': '<|im_start|>', 'im_end': '<|im_end|>',
    'system': '<|system|>', 'user': '<|human|>', 'model': '<|model|>',
    'tool': '<|tool_response|>',
    'thought_start': '<|thought_start|>', 'thought_end': '<|thought_end|>',
    'effort_low': '<|effort_low|>', 'effort_high': '<|effort_high|>', 'effort_max': '<|effort_max|>',
    'tool_call_start': '<|tool_call_start|>', 'tool_call_end': '<|tool_call_end|>',
    'tool_name_divider': '<|tool_name_divider|>',
    'fim_pre': '<|fim_prefix|>', 'fim_mid': '<|fim_middle|>', 'fim_suf': '<|fim_suffix|>',
    'pad': '<|pad|>',
}

#: Fallback template of this library, used when the tokenizer carries no chat template.
TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tokenizer', 'template.j2'
)

#: Thinking-effort alias table, kept in sync with `render_effort` in template.j2.
EFFORT_ALIAS = {
    'minimal': 'low', 'low': 'low',
    'medium': 'high', 'high': 'high', 'xhigh': 'high',
    'max': 'max', 'ultra': 'max',
}
EFFORT_DEFAULT = 'high'


def resolve_effort(effort: Optional[str] = None) -> str:
    '''
    Collapse a thinking-effort level name to its canonical level
    ('medium' -> 'high').

    Args:
        effort (Optional[str]): Requested level; None uses `EFFORT_DEFAULT`.

    Returns:
        str: The canonical level ('low', 'high' or 'max').

    Raises:
        ValueError: If the level is not one of the known names.
    '''
    level = str(effort or EFFORT_DEFAULT).strip().lower()
    if level not in EFFORT_ALIAS:
        raise ValueError(
            f'unexpected effort level: {effort!r} '
            f"(expected 'minimal', 'low', 'medium', 'high', 'xhigh', 'max' or 'ultra')"
        )
    return EFFORT_ALIAS[level]


def resolve_token_id(tokenizer: PackedTokenizer, name: str) -> Optional[int]:
    '''
    Resolve a logical name or the literal vocabulary text into a token id (None when
    the token is not in the vocabulary).

        resolve_token_id(tokenizer, 'im_end')          # -> logical name
        resolve_token_id(tokenizer, '<|im_end|>')      # -> literal vocabulary text

    Args:
        tokenizer (PackedTokenizer): Tokenizer to query.
        name (str): Logical name or literal token text.

    Returns:
        Optional[int]: The token id, or None when it is not in the vocabulary.
    '''
    return tokenizer.token_to_id(TOKEN_NAMES.get(name, name))


@dataclass
class Message:
    '''
    One encoded message: token ids plus the loss mask.

    Attributes:
        ids (list[int]): Token ids of the message.
        ignore_mask (list[bool]): True at positions excluded from the loss.
        role (Optional[str]): Canonical role of the message.
    '''

    ids: list[int]
    ignore_mask: list[bool] = field(default_factory=list)
    role: Optional[str] = None

    def __post_init__(self):
        '''
        Default the mask to "everything counts" and validate its length.

        Raises:
            ValueError: If `ignore_mask` is not as long as `ids`.
        '''
        if not self.ignore_mask:
            self.ignore_mask = [False] * len(self.ids)
        if len(self.ignore_mask) != len(self.ids):
            raise ValueError(
                f'ignore_mask length {len(self.ignore_mask)} != ids length {len(self.ids)}'
            )

    def __len__(self) -> int:
        '''
        Returns:
            int: Number of tokens in this message.
        '''
        return len(self.ids)

    def _resolve_range(self, begin: int, end: int, include: tuple[bool, bool]) -> tuple[int, int]:
        '''
        Clamp a range to the message and apply the boundary-inclusion flags.

        Args:
            begin (int): Start index.
            end (int): End index.
            include (tuple[bool, bool]): Whether to include the start and the end.

        Returns:
            tuple[int, int]: The half-open range `[lo, hi)` to operate on.
        '''
        lo = begin if include[0] else begin + 1
        hi = end + 1 if include[1] else end
        return max(0, lo), min(len(self.ids), hi)

    def mask_all(self) -> 'Message':
        '''
        Exclude the whole message from the loss.

        Returns:
            Message: This message, for chaining.
        '''
        self.ignore_mask = [True] * len(self.ids)
        return self

    def unmask_all(self) -> 'Message':
        '''
        Include the whole message in the loss.

        Returns:
            Message: This message, for chaining.
        '''
        self.ignore_mask = [False] * len(self.ids)
        return self

    def mask_between(
        self, begin: int, end: int,
        include_boundaries: tuple[bool, bool] = (False, False)
    ) -> 'Message':
        '''
        Exclude a range of positions from the loss.

        Args:
            begin (int): Start index.
            end (int): End index.
            include_boundaries (tuple[bool, bool]): Whether to include the start and
                the end positions themselves.

        Returns:
            Message: This message, for chaining.
        '''
        lo, hi = self._resolve_range(begin, end, include_boundaries)
        for i in range(lo, hi):
            self.ignore_mask[i] = True
        return self

    def unmask_between(
        self, begin: int, end: int,
        include_boundaries: tuple[bool, bool] = (False, False)
    ) -> 'Message':
        '''
        Include a range of positions in the loss.

        Args:
            begin (int): Start index.
            end (int): End index.
            include_boundaries (tuple[bool, bool]): Whether to include the start and
                the end positions themselves.

        Returns:
            Message: This message, for chaining.
        '''
        lo, hi = self._resolve_range(begin, end, include_boundaries)
        for i in range(lo, hi):
            self.ignore_mask[i] = False
        return self

    def insert_before(self, index: int, data: list[int], ignore_mask: bool = True) -> 'Message':
        '''
        Insert token ids before `index`, keeping ids and the mask aligned.

        Args:
            index (int): Position the new tokens are inserted at.
            data (list[int]): Token ids to insert.
            ignore_mask (bool): Mask value for the inserted tokens.

        Returns:
            Message: This message, for chaining.
        '''
        mask = [ignore_mask] * len(data) if isinstance(ignore_mask, bool) else list(ignore_mask)
        self.ids = self.ids[:index] + list(data) + self.ids[index:]
        self.ignore_mask = self.ignore_mask[:index] + mask + self.ignore_mask[index:]
        return self

    def insert_after(self, index: int, data: list[int], ignore_mask: bool = True) -> 'Message':
        '''
        Insert token ids right after `index`.

        Args:
            index (int): Position the new tokens are inserted after.
            data (list[int]): Token ids to insert.
            ignore_mask (bool): Mask value for the inserted tokens.

        Returns:
            Message: This message, for chaining.
        '''
        return self.insert_before(index + 1, data, ignore_mask)

    def find(self, token_id: int, start: int = 0) -> int:
        '''
        Find the first occurrence of a token.

        Args:
            token_id (int): Token to look for.
            start (int): Index to start searching at.

        Returns:
            int: The index, or -1 when the token does not occur.
        '''
        try:
            return self.ids.index(token_id, start)
        except ValueError:
            return -1

    def find_last(self, token_id: int) -> int:
        '''
        Find the last occurrence of a token.

        Args:
            token_id (int): Token to look for.

        Returns:
            int: The index, or -1 when the token does not occur.
        '''
        for i in range(len(self.ids) - 1, -1, -1):
            if self.ids[i] == token_id:
                return i
        return -1


class Session:
    '''
    A conversation being built: append messages, then package them into tensors.

    Messages are encoded with this library's chat template and the per-role mask
    policy decides which tokens contribute to the loss.

    Attributes:
        tokenizer (PackedTokenizer): Tokenizer carrying the chat template.
        messages (list[Message]): The encoded messages.
        policy (dict[str, MaskPolicy]): Default mask policy per role.
    '''

    _ROLE_ALIAS = {
        'assistant': 'model',
        'developer': 'system',
        'human': 'user',
    }

    def __init__(self, tokenizer: PackedTokenizer):
        '''
        Build a session and resolve every special token of the vocabulary.

        Args:
            tokenizer (PackedTokenizer): Tokenizer that already carries this library's
                chat template (template.j2); when it has none, the bundled one is
                installed automatically.
        '''
        self.tokenizer = tokenizer
        self.messages: list[Message] = []

        self._tokens = dict(TOKEN_NAMES)
        self._ids: dict[str, Optional[int]] = {
            k: tokenizer.token_to_id(v) for k, v in self._tokens.items()
        }
        self._ensure_template()

        self.policy: dict[str, MaskPolicy] = {
            'system': 'all',
            'user': 'all',
            'tool': 'all',
            'model': 'content',
            'fim': 'fim',
        }

    def _ensure_template(self) -> None:
        '''
        Install the bundled template.j2 when the tokenizer carries no chat template.
        '''
        if (getattr(self.tokenizer, 'template', '') or '').strip():
            return
        with open(TEMPLATE_PATH, encoding='utf-8') as f:
            self.tokenizer.set_chat_template(f.read())

    @property
    def tokens(self) -> dict:
        '''
        Returns:
            dict: A copy of the logical name -> vocabulary token text mapping.
        '''
        return dict(self._tokens)

    def token_id(self, name: str) -> Optional[int]:
        '''
        Resolve a logical name ('im_end') or the literal vocabulary text
        ('<|im_end|>') into a token id (None when it is not in the vocabulary).

        Args:
            name (str): Logical name or literal token text.

        Returns:
            Optional[int]: The token id, or None.
        '''
        if name in self._ids:
            return self._ids[name]
        return self.tokenizer.token_to_id(name)

    def _id(self, name: str) -> int:
        '''
        Resolve a logical name into a token id.

        Args:
            name (str): Logical token name.

        Returns:
            int: The token id.

        Raises:
            ValueError: If the token is not in the vocabulary.
        '''
        tid = self.token_id(name)
        if tid is None:
            raise ValueError(f'token {self._tokens.get(name, name)!r} not found in vocab')
        return tid

    def special_ids(self) -> set:
        '''
        Returns:
            set: The ids of the special tokens that could be resolved, used to filter
            structural tokens out of generated text.
        '''
        return {tid for tid in self._ids.values() if tid is not None}

    def set_token(self, mapping: dict) -> 'Session':
        '''
        Override or extend the logical name -> token text mapping.

        Args:
            mapping (dict): Logical name to token text.

        Returns:
            Session: This session, for chaining.
        '''
        self._tokens.update(mapping)
        self._ids.update({k: self.tokenizer.token_to_id(v) for k, v in mapping.items()})
        return self

    def set_policy(self, role: str, policy: MaskPolicy) -> 'Session':
        '''
        Set the default mask policy of one role.

        Args:
            role (str): Role name; aliases such as 'assistant' are normalised.
            policy (MaskPolicy): Policy name or an explicit boolean mask.

        Returns:
            Session: This session, for chaining.
        '''
        self.policy[self._ROLE_ALIAS.get(role, role)] = policy
        return self

    def _encode_message(self, message: dict, tools: Optional[list] = None) -> list[int]:
        '''
        Render one message through the chat template.

        Args:
            message (dict): The message to encode.
            tools (Optional[list]): Tool (function) declarations, rendered by the
                template in its leading system block. Pass them with the **first**
                message of a conversation: the template emits that block in front of
                the conversation, so handing them over per message would repeat it.

        Returns:
            list[int]: The token ids of that message.
        '''
        msg = copy.deepcopy(message)
        role = self._ROLE_ALIAS.get(msg.get('role', 'user'), msg.get('role', 'user'))
        msg['role'] = role
        if role == 'model':
            msg.setdefault('thought', None)
            msg.setdefault('reasoning_content', None)

        extra = {'tools': list(tools)} if tools else {}

        return self.tokenizer.apply_chat_template(
            [msg],
            add_generation_prompt=False,
            tokenize=True,
            **extra,
        )

    def _apply_policy(self, msg: Message, policy: MaskPolicy) -> None:
        '''
        Apply a mask policy to one message, in place.

        Args:
            msg (Message): The message to mask.
            policy (MaskPolicy): A policy name, or an explicit boolean mask whose
                length must match the message.

        Raises:
            ValueError: If an explicit mask has the wrong length, or the policy name
                is unknown.
        '''
        if isinstance(policy, (list, tuple)):
            if len(policy) != len(msg):
                raise ValueError(f'explicit mask length {len(policy)} != message length {len(msg)}')
            msg.ignore_mask = [bool(x) for x in policy]
            return

        if policy == 'all':
            msg.mask_all()
        elif policy == 'none':
            msg.unmask_all()
        elif policy == 'content':
            msg.mask_all()
            anchor = self._ids.get('thought_start')
            idx = msg.find(anchor) if anchor is not None else -1
            if idx < 0:
                role_id = self._ids.get(msg.role or '')
                idx = msg.find(role_id) if role_id is not None else -1
                if idx >= 0:
                    idx += 1
            if idx >= 0:
                msg.unmask_between(idx, len(msg), include_boundaries=(True, True))
        elif policy == 'thought':
            msg.mask_all()
            s_id, e_id = self._ids.get('thought_start'), self._ids.get('thought_end')
            if s_id is None or e_id is None:
                return
            s = msg.find(s_id)
            e = msg.find(e_id, max(s, 0))
            if s >= 0 and e > s:
                msg.unmask_between(s, e, include_boundaries=(False, False))
        elif policy == 'answer':
            msg.mask_all()
            cot_end_id, im_end_id = self._ids.get('thought_end'), self._ids.get('im_end')
            cot_end_idx = msg.find(cot_end_id) if cot_end_id is not None else -1
            im_end_idx = msg.find_last(im_end_id) if im_end_id is not None else -1
            if im_end_idx >= 0:
                model_id = self._ids.get('model')
                fallback_idx = msg.find(model_id) if model_id is not None else 0
                start_idx = cot_end_idx if cot_end_idx >= 0 else fallback_idx
                if start_idx >= 0 and im_end_idx > start_idx:
                    msg.unmask_between(start_idx, im_end_idx, include_boundaries=(False, True))
        elif policy == 'fim':
            msg.mask_all()
            mid_id, end_id = self._ids.get('fim_mid'), self._ids.get('im_end')
            if mid_id is None or end_id is None:
                return
            mid_idx = msg.find(mid_id)
            end_idx = msg.find_last(end_id)
            if mid_idx >= 0 and end_idx > mid_idx:
                msg.unmask_between(mid_idx, end_idx, include_boundaries=(True, True))
        else:
            raise ValueError(f'unknown mask policy: {policy!r}')

    def _pop_gen_prompt(self) -> Optional[Message]:
        '''
        Remove the generation prompt if it is the last message.

        Returns:
            Optional[Message]: The generation prompt that was removed, or None.
        '''
        if self.messages and self.messages[-1].role == '__gen_prompt__':
            return self.messages.pop()
        return None

    def _append(self, msg: Message) -> None:
        '''
        Append a message, keeping a generation prompt last.

        Args:
            msg (Message): The message to append.
        '''
        gen = self._pop_gen_prompt()
        self.messages.append(msg)
        if gen is not None:
            self.messages.append(gen)

    def add_message(
        self,
        message: dict,
        mask: Optional[MaskPolicy] = None,
        tools: Optional[list] = None,
    ) -> 'Session':
        '''
        Encode and append one message; when `mask` is None the policy of its role is
        used.

        Args:
            message (dict): The message to append.
            mask (Optional[MaskPolicy]): Mask policy override.
            tools (Optional[list]): Tool (function) declarations rendered into the
                template's leading system block; pass them with the first message of
                a conversation.

        Returns:
            Session: This session, for chaining.
        '''
        if isinstance(message.get('tool_calls'), str):
            message['tool_calls'] = json.loads(message['tool_calls'])
        role = self._ROLE_ALIAS.get(message.get('role', 'user'), message.get('role', 'user'))

        msg = Message(ids=self._encode_message(message, tools=tools), role=role)
        self._apply_policy(msg, mask if mask is not None else self.policy.get(role, 'all'))

        self._append(msg)
        return self

    def add_messages(self, messages: list[dict], tools: Optional[list] = None) -> 'Session':
        '''
        Append a list of messages in order.

        Args:
            messages (list[dict]): The messages to append.
            tools (Optional[list]): Tool (function) declarations; they are rendered in
                the leading system block, so they are handed to the first message.

        Returns:
            Session: This session, for chaining.
        '''
        for index, message in enumerate(messages):
            self.add_message(message, tools=tools if index == 0 else None)
        return self

    def add_generation_prompt(
        self,
        enable_thinking: bool = False,
        disable_thinking: bool = False,
        effort: Optional[str] = None,
        enable_think: bool = True,
        enable_effort: bool = True,
    ) -> 'Session':
        '''
        Append the generation prompt `<|im_start|><|model|>`, which always stays after
        the last real message.

        Args:
            enable_thinking (bool): Open a reasoning section by appending the effort
                token after `<|thought_start|>`. False (the default) writes no reasoning
                section at all - callers that want the empty
                `<|thought_start|><|thought_end|>` section ask for it with
                `disable_thinking=True` (or `enable_effort=False`), and callers that
                want the effort pre-filled ask for `enable_thinking=True`.
            disable_thinking (bool): Emit an empty reasoning section
                `<|thought_start|><|thought_end|>` without an effort token.
            effort (Optional[str]): Thinking effort, one of 'minimal' / 'low' /
                'medium' / 'high' / 'xhigh' / 'max' / 'ultra', collapsed onto three
                tokens: minimal|low -> `<|effort_low|>`,
                medium|high|xhigh -> `<|effort_high|>`, max|ultra -> `<|effort_max|>`.
                None means the default level `EFFORT_DEFAULT`.
            enable_think (bool): Keep the reasoning section of the template
                (`<|thought_start|>` / `<|thought_end|>`) at all; False drops it, so
                the prompt is only `<|im_start|><|model|>` - the same switch the
                template knows as `enable_think`.
            enable_effort (bool): Emit the effort token inside the section; False
                leaves an empty `<|thought_start|><|thought_end|>` - the same switch
                the template knows as `enable_effort`.

        Returns:
            Session: This session, for chaining.

        Raises:
            ValueError: If `enable_thinking` and `disable_thinking` are both True.
        '''
        if enable_thinking and disable_thinking:
            raise ValueError('enable_thinking and disable_thinking cannot both be True')
        self._pop_gen_prompt()

        keys = ['im_start', 'model']
        if not enable_think:
            pass
        elif enable_thinking:
            if enable_effort:
                keys += ['thought_start', f'effort_{resolve_effort(effort)}']
            else:
                keys += ['thought_start', 'thought_end']
        elif disable_thinking or not enable_effort:
            keys += ['thought_start', 'thought_end']

        msg = Message(ids=[self._id(key) for key in keys], role='__gen_prompt__')
        msg.mask_all()
        self.messages.append(msg)
        return self

    def pop(self, index: int = -1) -> Message:
        '''
        Remove and return one message.

        Args:
            index (int): Index of the message; -1 is the last one.

        Returns:
            Message: The removed message.
        '''
        return self.messages.pop(index)

    def clear(self) -> 'Session':
        '''
        Remove every message.

        Returns:
            Session: This session, for chaining.
        '''
        self.messages.clear()
        return self

    def pad_to(self, length: int, side: Literal['left', 'right'] = 'right') -> 'Session':
        '''
        Append one `<|pad|>` message on the left or the right to bring the total
        conversation length up to `length`.

        Args:
            length (int): Target total length.
            side (Literal['left', 'right']): Which side the padding message goes to.

        Returns:
            Session: This session, for chaining.

        Raises:
            ValueError: If `side` is neither 'left' nor 'right'.
        '''
        if len(self) >= length:
            return self
        if side not in ('left', 'right'):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        delta = length - len(self)
        pad_msg = Message(
            ids=[self._id('pad')] * delta,
            ignore_mask=[True] * delta,
            role='__padding__',
        )
        self.messages.insert(0 if side == 'left' else len(self.messages), pad_msg)
        return self

    @property
    def input_ids(self) -> list[int]:
        '''
        Returns:
            list[int]: The token ids of the whole conversation.
        '''
        return [tid for m in self.messages for tid in m.ids]

    @property
    def ignore_mask(self) -> list[bool]:
        '''
        Returns:
            list[bool]: The concatenated loss mask of the whole conversation.
        '''
        return [flag for m in self.messages for flag in m.ignore_mask]

    @property
    def labels(self) -> list[int]:
        '''
        Returns:
            list[int]: `input_ids` with -100 at every masked position.
        '''
        return [-100 if flag else tid for tid, flag in zip(self.input_ids, self.ignore_mask)]

    def to_tensors(
        self,
        device: Union[str, torch.device] = 'cpu',
        pad_to: Optional[int] = None,
        batch_dim: bool = False,
    ) -> dict:
        '''
        Package the conversation into the tensor dict used for training / inference.

        Args:
            device (Union[str, torch.device]): Device the tensors are placed on.
            pad_to (Optional[int]): Length the sequence is padded to (padding is masked
                out of the loss and from the attention).
            batch_dim (bool): Add a leading batch dimension of 1 to every tensor field.

        Returns:
            dict: The packaged tensors.
        '''
        ids = self.input_ids
        labels = self.labels
        attention = [1] * len(ids)

        if pad_to is not None and len(ids) < pad_to:
            delta = pad_to - len(ids)
            ids += [self.token_id('pad') or 0] * delta
            labels += [-100] * delta
            attention += [0] * delta

        to_tensor = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
        out = {
            'input_ids': to_tensor(ids),
            'labels': to_tensor(labels),
            'attention_mask': to_tensor(attention),
        }

        if batch_dim:
            for key in ('input_ids', 'labels', 'attention_mask'):
                out[key] = out[key].unsqueeze(0)

        return out

    def decode(self, skip_special_tokens: bool = False) -> str:
        '''
        Decode the whole conversation.

        Args:
            skip_special_tokens (bool): Drop the special tokens from the output.

        Returns:
            str: The decoded text.
        '''
        return self.tokenizer.decode(self.input_ids, skip_special_tokens=skip_special_tokens)

    def __len__(self) -> int:
        '''
        Returns:
            int: Total number of tokens in the conversation.
        '''
        return sum(len(m) for m in self.messages)

    def __iter__(self) -> Iterator[Message]:
        '''
        Returns:
            Iterator[Message]: Iterator over the messages.
        '''
        return iter(self.messages)

    def __getitem__(self, index: int) -> Message:
        '''
        Args:
            index (int): Message index.

        Returns:
            Message: The message at that index.
        '''
        return self.messages[index]
