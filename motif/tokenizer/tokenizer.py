import json
import os
import zipfile
import copy

from dataclasses import dataclass
from tokenizers  import Tokenizer, pre_tokenizers, decoders, normalizers
from tokenizers.models   import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

from typing import Any, Dict, Iterable, List, Optional, Union

from motif.tokenizer.spec_token import token, SPEC_TOKEN

RESERVED_ENTRIES = (
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
    'safe_rules.json',
    'safe_escape.txt',
)


def find_entry(names: List[str], name: str) -> Optional[str]:
    '''
    Find an archive entry by its base name, ignoring any directory prefix.

    Args:
        names (List[str]): Every entry name of the archive.
        name (str): Base name to look for.

    Returns:
        Optional[str]: The full entry name, or None when the archive has no such entry.
    '''
    for entry in names:
        if entry == name or entry.endswith(f'/{name}'):
            return entry
    return None


@dataclass
class TokenizerTrainerResult:
    '''
    Result of the tokenizer trainer creation.

    Attributes:
        tokenizer (Tokenizer): The configured tokenizer instance.
        trainer (BpeTrainer): The configured BPE trainer instance.
    '''
    tokenizer: Tokenizer
    trainer: BpeTrainer

    def train_from_iterator(self, iterable: Iterable[str]) -> 'TokenizerTrainerResult':
        '''
        Train the BPE model on an iterator of texts.

        Args:
            iterable (Iterable[str]): Iterator yielding the training texts.

        Returns:
            TokenizerTrainerResult: This result, for chaining.
        '''
        self.tokenizer.train_from_iterator(iterable, self.trainer)
        return self

    @property
    def packed_tokenizer(self) -> 'PackedTokenizer':
        '''
        Returns:
            PackedTokenizer: A `PackedTokenizer` wrapping the trained tokenizer.
        '''
        return PackedTokenizer(
            tokenizer=self.tokenizer
        )


def create_tokenizer_trainer(
    unk_token: str=token('unk'),
    vocab_size: int=5*(2**13),
    special_tokens: list[str]=SPEC_TOKEN,
    use_norm: bool = True,
    use_nfc: bool = True
) -> TokenizerTrainerResult:
    '''
    Creates a BPE Tokenizer trainer.

    Configures and returns a tokenizer trainer object for training BPE (Byte-Pair
    Encoding) models. The trainer is pre-configured with NFC normalization, digit
    splitting, and byte-level pre-tokenization.

    Args:
        unk_token (str): Identifier for unknown tokens. Defaults to '<|unk|>'.
        vocab_size (int): Target vocabulary size. Defaults to 5*(2**13).
        special_tokens (list[str]): List of special tokens. Defaults to SPEC_TOKEN,
            which covers the basic, chat, role, reasoning, tool, FIM, multimodal, plan,
            action, grounding and object-reference groups.
        use_norm (bool): Whether to attach a normalizer at all. Defaults to True.
        use_nfc (bool): Whether to use NFC rather than NFKC normalization; ignored when
            `use_norm` is False. Defaults to True.

    Returns:
        TokenizerTrainerResult: A dataclass containing the tokenizer and trainer instances.
    '''
    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    if use_norm:
        tokenizer.normalizer = normalizers.NFC() if use_nfc else normalizers.NFKC()

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True
        )
    ])

    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=32,
        min_frequency=10
    )

    return TokenizerTrainerResult(tokenizer=tokenizer, trainer=trainer)


class PackedTokenizer:
    '''
    The library's tokenizer wrapper: chat template, special tokens, safe escaping and
    a single-file archive format.

    The wrapper can be built around a trained `tokenizers.Tokenizer` or loaded from a
    zip archive written by `save`.

    Attributes:
        config (Dict[str, Any]): Token configuration (`unk_token`, `pad_token`,
            `bos_token`, `eos_token`).
        template (str): The chat template (Jinja source).
        extra_files (Dict[str, bytes]): Extra files packed into the archive.
        safe_escape (str): Token text used to escape the structural characters.
        safe_escape_id (Optional[int]): Cached id of that token.
        safe_rules (Dict[str, str]): Character -> escaped replacement mapping.
    '''

    def __init__(
        self,
        tokenizer: Optional[Union[Tokenizer, str]] = None,
        safe_escape: str = token('safe_escape'),
        safe_rules: Optional[Dict[str, str]] = None
    ):
        '''Wrap a trained tokenizer, or load one from an archive written by `save`.

        Args:
            tokenizer (Optional[Union[Tokenizer, str]]): A trained tokenizer, or the path
                of an archive to load. Defaults to None, leaving the wrapper empty.
            safe_escape (str): Token text used to escape structural characters. Defaults
                to '<|safe_escape|>'.
            safe_rules (Optional[Dict[str, str]]): Character -> escaped replacement
                mapping, whose values may contain the `{safe}` placeholder. Defaults to
                escaping `]`, `[` and `|`.
        '''
        self._tokenizer: Optional[Tokenizer] = None
        self._fast_tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.config = {}
        self.template = ''

        self.extra_files: Dict[str, bytes] = {}

        self.safe_escape = safe_escape
        self.safe_escape_id: Optional[int] = None

        self.safe_rules: Dict[str, str] = safe_rules or {
            ']': '{safe}]',
            '[': '[{safe}',
            '|': '{safe}|{safe}'
        }

        if isinstance(tokenizer, str):
            self.load(tokenizer)
        elif isinstance(tokenizer, Tokenizer):
            self._tokenizer = tokenizer
            self._update_fast_tokenizer()

    def _update_fast_tokenizer(self) -> None:
        '''
        Rebuild the cached `PreTrainedTokenizerFast` wrapper.

        Does nothing while no `tokenizers.Tokenizer` is loaded.
        '''
        if self._tokenizer is None:
            self._fast_tokenizer = None
            return

        self._fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer,
            unk_token=self.config.get('unk_token', token('unk')),
            pad_token=self.config.get('pad_token', token('pad')),
            bos_token=self.config.get('bos_token', token('bos')),
            eos_token=self.config.get('eos_token', token('eos')),
            chat_template=self.template,
            clean_up_tokenization_spaces=False
        )

    def set_chat_template(self, template: str) -> 'PackedTokenizer':
        '''
        Install the chat template used by `apply_chat_template`.

        Args:
            template (str): Jinja source of the template.

        Returns:
            PackedTokenizer: This tokenizer, for chaining.

        Raises:
            TypeError: If `template` is not a string.
        '''
        if not isinstance(template, str):
            raise TypeError(f'template must be str, got {type(template).__name__}')
        self.template = template
        self._update_fast_tokenizer()
        return self

    def set_eos(self, eos: str) -> bool:
        '''
        Set the end-of-sequence token, if the vocabulary has it.

        Args:
            eos (str): Token text of the end-of-sequence token.

        Returns:
            bool: True when the token exists and was set; False otherwise.
        '''
        if self._tokenizer.token_to_id(eos) is not None:
            self.config['eos_token'] = eos
            return True
        return False

    @property
    def token_eos(self) -> Optional[str]:
        '''
        Returns:
            Optional[str]: Token text of the configured end-of-sequence token, or the
            default `<|eos|>`.
        '''
        return self.config.get('eos_token', token('eos'))

    def set_pad(self, pad: str) -> bool:
        '''
        Set the padding token, if the vocabulary has it.

        Args:
            pad (str): Token text of the padding token.

        Returns:
            bool: True when the token exists and was set; False otherwise.
        '''
        if self._tokenizer.token_to_id(pad) is not None:
            self.config['pad_token'] = pad
            return True
        return False

    @property
    def token_pad(self) -> Optional[str]:
        '''
        Returns:
            Optional[str]: Token text of the configured padding token, or the default
            `<|pad|>`.
        '''
        return self.config.get('pad_token', token('pad'))

    @property
    def tokenizer(self) -> Tokenizer:
        '''
        Returns:
            Tokenizer: The underlying `tokenizers.Tokenizer`.

        Raises:
            ValueError: If no tokenizer has been loaded yet.
        '''
        if self._tokenizer is None:
            raise ValueError('Tokenizer is not loaded.')
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        '''
        Returns:
            int: Size of the vocabulary.
        '''
        return self._tokenizer.get_vocab_size()

    @property
    def fast_tokenizer(self) -> PreTrainedTokenizerFast:
        '''
        Returns:
            PreTrainedTokenizerFast: The HuggingFace wrapper used for chat templates
            and encoding.

        Raises:
            ValueError: If no tokenizer has been loaded yet.
        '''
        if self._fast_tokenizer is None:
            raise ValueError('Tokenizer is not loaded.')
        return self._fast_tokenizer

    def token_to_id(self, token: str) -> Optional[int]:
        '''
        Resolve a token id.

        Args:
            token (str): Token text.

        Returns:
            Optional[int]: The token id, or None when the vocabulary has no such
            token.
        '''
        return self._tokenizer.token_to_id(token)

    def ensure_escape(self) -> int:
        '''
        Resolve (and cache) the id of the safe-escape token.

        Returns:
            int: The id of `safe_escape`.

        Raises:
            ValueError: If the vocabulary has no such token.
        '''
        if self.safe_escape_id is None:
            tid = self._tokenizer.token_to_id(self.safe_escape)
            if tid is None:
                raise ValueError(f'Escape token {self.safe_escape} not found in vocab.')
            self.safe_escape_id = tid
        return self.safe_escape_id

    def _apply_safe_rules(self, text: str) -> str:
        '''
        Escape the structural characters of `text` with the safe-escape token.

        Args:
            text (str): Raw text.

        Returns:
            str: The text with every rule of `safe_rules` applied.
        '''
        for old, new in self.safe_rules.items():
            text = text.replace(old, new.format(safe=self.safe_escape))
        return text

    def _sanitize_content(self, content: Any) -> Any:
        '''
        Apply the safe rules to a message content, which may be a string or a list of
        typed parts.

        Args:
            content (Any): The content field of a message.

        Returns:
            Any: The same shape, with the text of every text part escaped.
        '''
        if isinstance(content, str):
            return self._apply_safe_rules(content)
        elif isinstance(content, list):
            return [
                {**item, 'text': self._sanitize_content(item['text'])} if item.get('type') == 'text' else item
                for item in content
            ]
        return content

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
        **kwargs
    ) -> List[int]:
        '''
        Render messages through the chat template and return the token ids.

        The content of every message is deep-copied and escaped first, and the escape
        token is filtered out of the result, so the structural characters of the text
        can never be confused with the template's own markers. Model messages get a
        'thought' default of None, so the template falls through to
        'reasoning_content' whenever no 'thought' was given.

        Args:
            messages (List[Dict[str, Any]]): The conversation.
            add_generation_prompt (bool): Append the assistant generation prompt.
            **kwargs: Extra keyword arguments for the template (for example the
                modality capability switches).

        Returns:
            List[int]: The token ids of the rendered conversation.

        Raises:
            ValueError: If the vocabulary has no safe-escape token.
        '''
        escape_id = self.ensure_escape()

        safe_messages = copy.deepcopy(messages)
        for msg in safe_messages:
            if 'content' in msg:
                msg['content'] = self._sanitize_content(msg['content'])
            if isinstance(msg, dict) and msg.get('role') in ('model', 'assistant'):
                msg.setdefault('thought', None)

        kwargs.pop('tokenize', None)

        raw = self.fast_tokenizer.apply_chat_template(
            safe_messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **kwargs
        )

        if isinstance(raw, dict) or hasattr(raw, 'input_ids'):
            raw = raw['input_ids']
        if raw and isinstance(raw[0], list):
            raw = raw[0]

        return [tid for tid in raw if tid != escape_id]

    def encode(self, text: str, add_special_tokens: bool = False, apply_safe_rule: bool = True, **kwargs) -> List[int]:
        '''
        Encode raw text into token ids.

        Args:
            text (str): The text to encode.
            add_special_tokens (bool): Let the tokenizer add its special tokens.
            apply_safe_rule (bool): Escape the structural characters first.
            **kwargs: Forwarded to the underlying tokenizer.

        Returns:
            List[int]: The token ids, with the escape token filtered out.
        '''
        escape_id = self.ensure_escape()

        safe_text = self._apply_safe_rules(text) if apply_safe_rule else text

        raw_ids = self.fast_tokenizer.encode(safe_text, add_special_tokens=add_special_tokens, **kwargs)

        return [tid for tid in raw_ids if tid != escape_id]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        '''
        Decode token ids back into text.

        Args:
            token_ids (List[int]): The ids to decode.
            skip_special_tokens (bool): Drop the special tokens from the output.

        Returns:
            str: The decoded text.
        '''
        return self.fast_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def add_file(self, name: str, data: Union[str, bytes]) -> 'PackedTokenizer':
        '''
        Attach an extra file to be written by `save`.

        Args:
            name (str): Name inside the archive; must not collide with a reserved
                name.
            data (Union[str, bytes]): File content; a string is encoded as UTF-8.

        Returns:
            PackedTokenizer: This tokenizer, for chaining.
        '''
        if isinstance(data, str):
            data = data.encode('utf-8')
        self.extra_files[name] = data
        return self

    def get_file(self, name: str) -> Optional[bytes]:
        '''
        Args:
            name (str): Name of an extra file.

        Returns:
            Optional[bytes]: Its content, or None when it is not attached.
        '''
        return self.extra_files.get(name)

    def save(self, path: str) -> 'PackedTokenizer':
        '''
        Write the tokenizer, its configuration, the chat template, the safe rules and
        every extra file into one zip archive.

        Args:
            path (str): Destination archive.

        Returns:
            PackedTokenizer: This tokenizer, for chaining.

        Raises:
            ValueError: If there is no tokenizer to save, or an extra file uses a
                reserved name.
        '''
        if self._tokenizer is None:
            raise ValueError('No tokenizer to save.')

        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('tokenizer.json', self._tokenizer.to_str())
            z.writestr('tokenizer_config.json', json.dumps(self.config, indent=2))
            z.writestr('chat_template.jinja', self.template)
            z.writestr('safe_rules.json', json.dumps(self.safe_rules, indent=2))
            z.writestr('safe_escape.txt', self.safe_escape)

            for name, data in self.extra_files.items():
                if name in RESERVED_ENTRIES:
                    raise ValueError(f'Cannot use reserved file name for extra files: {name}')
                z.writestr(name, data)

        return self

    def load(self, path: str) -> 'PackedTokenizer':
        '''
        Load a tokenizer written by `save`, and rebuild the fast tokenizer wrapper.

        Reserved entries are read into the matching attributes, every other entry of
        the archive lands in `extra_files`.

        Args:
            path (str): The archive written by `save`.

        Returns:
            PackedTokenizer: This tokenizer, for chaining.

        Raises:
            FileNotFoundError: If the archive does not exist.
            ValueError: If the archive has no `tokenizer.json` entry.
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')

        with zipfile.ZipFile(path, 'r') as z:
            entries = z.namelist()

            tokenizer_file = find_entry(entries, 'tokenizer.json')
            if tokenizer_file is None:
                raise ValueError('tokenizer.json not found in zip file')
            self._tokenizer = Tokenizer.from_str(z.read(tokenizer_file).decode('utf-8'))

            config_file = find_entry(entries, 'tokenizer_config.json')
            if config_file:
                self.config = json.loads(z.read(config_file).decode('utf-8'))

            template_file = find_entry(entries, 'chat_template.jinja')
            if template_file:
                self.template = z.read(template_file).decode('utf-8')

            rules_file = find_entry(entries, 'safe_rules.json')
            if rules_file:
                self.safe_rules = json.loads(z.read(rules_file).decode('utf-8'))

            escape_file = find_entry(entries, 'safe_escape.txt')
            if escape_file:
                self.safe_escape = z.read(escape_file).decode('utf-8')
                self.safe_escape_id = None

            self.extra_files = {}
            for entry in entries:
                base_name = entry.split('/')[-1]
                if base_name and base_name not in RESERVED_ENTRIES:
                    self.extra_files[entry] = z.read(entry)

        self._update_fast_tokenizer()
        return self
