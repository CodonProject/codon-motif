import bisect
import html
import os
import random
import re
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow.parquet as pq
from tqdm import tqdm

from motif.data.base import MotifDataset
from motif.tokenizer.tokenizer import PackedTokenizer


#: Supported `dataset_type` values; 'none' = use the text as it is.
DATASET_TYPES = ('none', 'tiny_corpus')

#: CJK ideographs (basic block + extensions A/B + compatibility block).
_CJK = r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff'
#: CJK punctuation and full-width symbols (the ranges below).
_CJK_PUNCT = (r'\u3000-\u303f'
              r'\uff01-\uff0f'
              r'\uff1a-\uff20'
              r'\uff3b-\uff40'
              r'\uff5b-\uff65')
_CJK_CLASS = f'[{_CJK}{_CJK_PUNCT}]'
#: A space between two CJK characters (punctuation included) - only that is eaten,
#: the spaces between CJK and Latin are left alone.
_CJK_SPACE = re.compile(
    rf'(?<={_CJK_CLASS})[ \t\u3000]+(?={_CJK_CLASS})'
)
#: Shape of an HTML escape entity (the ones that cannot be recognised are deleted):
#: &quot; / &lt; / &nbsp; / &#39; / &#x27; ...
#: The semicolon is required, so ordinary text such as "AT&T" or "R&D" is not
#: damaged.
_HTML_ENTITY = re.compile(
    r'&(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6});'
)
#: The various Unicode spaces (nbsp / ideographic space / narrow space ...) are
#: normalised to a plain space first.
_UNICODE_SPACE = re.compile(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]')
#: Runs of spaces / tabs (decoding entities often leaves double spaces behind).
_SPACE_RUN = re.compile(r'[ \t\f\v]{2,}')

#: Space + closing punctuation -> glue back: `no ,` -> `no,`, `swans .` -> `swans.`
_DETOK_BEFORE = re.compile(r'[ \t]+([,\.!?;:%\)\]\}…])')
#: Opening bracket + space -> glue back: `( 1` -> `(1)`
_DETOK_AFTER = re.compile(r'([\(\[\{])[ \t]+')
#: Space + contraction suffix -> glue back: `it 's` / `do n't` / `I 'm`
_CONTRACTION = re.compile(r"[ \t]+('(?:s|t|m|re|ve|ll|d)\b|n't\b)", re.IGNORECASE)
#: Moses' hyphen escape: `fast @-@ acting` -> `fast-acting` (spaces on both sides are eaten)
_MOSES_HYPHEN = re.compile(r'[ \t]*@-@[ \t]*')

#: A run of dashes: `--` / `---` (tight), or `- -` / `-  -` / `- - -` (spaced out).
#: **The dashes in the Chinese side of this corpus are actually written `- -`**
#: (measured on wmt_zh_en/5: `很 精彩 - - 我只 对 …`), so runs of hyphens with spaces
#: in between have to count as well; a single `-` does not (it looks more like a range
#: or a coordination).
_DASH_RUN = r'(?:-{2,}|-+(?:[ \t]+-+)+)'
#: A `--` / `- -` style dash next to CJK (on either side) becomes the CJK dash `——`,
#: with the spaces on both sides eaten.
_CJK_DASH = re.compile(
    rf'(?:(?<={_CJK_CLASS})[ \t]*{_DASH_RUN}[ \t]*(?={_CJK_CLASS})'
    rf'|(?<={_CJK_CLASS})[ \t]*{_DASH_RUN}'
    rf'|[ \t]*{_DASH_RUN}[ \t]*(?={_CJK_CLASS}))'
)
#: Used by `dash_scope='row'`: as long as the line contains CJK, every dash run of
#: that line is replaced (the spaces are kept).
_ANY_DASH = re.compile(_DASH_RUN)
#: Tests whether "this line contains CJK".
_HAS_CJK = re.compile(f'[{_CJK}]')
#: Values of `clean_tiny_corpus(dash_scope=...)`.
DASH_SCOPES = ('cjk', 'row', 'none')


def detokenize_english(text: str) -> str:
    '''
    Glue Moses-tokenized English back together (the English side of this corpus looks
    like `no , there 's too much ... it .`).

    * `it 's` / `do n't` / `I 'm` -> `it's` / `don't` / `I'm` (including the ones that
      came out of an `&apos;` entity);
    * space + closing punctuation -> glued back (`, . ! ? ; : % ) ] }`);
    * opening bracket + space -> glued back (`( [ {`);
    * `@-@` -> `-` (Moses' hyphen escape).

    **Deliberately left alone**:

    * **spaces inside or outside quotes** (`said " hello "`): this corpus is truncated
      per line, so the number of quotes on a line can be odd and deciding "opening
      vs closing" by pairing is unreliable; gluing the wrong direction turns
      `said " hello` into `said" hello` or `hello " to` into `hello "to`. Doing it
      properly needs a pairing heuristic whose benefit does not justify the risk.
    * a hyphen with spaces on both sides such as `2 - 7` (it may be a range or a
      footnote number);
    * full-width CJK punctuation, and the ordinary spaces between CJK and Latin or
      inside English.

    Args:
        text (str): The line to detokenize.

    Returns:
        str: The line with the English detokenization rules applied.
    '''
    if not text:
        return text
    text = _MOSES_HYPHEN.sub('-', text)
    text = _CONTRACTION.sub(r'\1', text)
    text = _DETOK_BEFORE.sub(r'\1', text)
    text = _DETOK_AFTER.sub(r'\1', text)
    return text


def clean_tiny_corpus(text: str, unescape: bool = True, detokenize: bool = True,
                      dash_scope: str = 'cjk') -> str:
    '''
    Body-text cleaning for TinyCorpus, in six steps:

    1. **Handle the HTML escape entities.** This corpus only has 7 kinds, 619 in
       total (sampled over 6001 lines): `&apos;` 378, `&quot;` 206, `&#93;` 13,
       `&#91;` 12, `&amp;` 7, `&gt;` 2, `&#124;` 1 - all standard escapes, and **not a
       single one is inside a word**.

       * `unescape=True` (the default): **restore** them to characters following the
         HTML standard, `&apos;` -> `'`, `&quot;` -> `"` (a doubly escaped
         `&amp;quot;` is restored all the way down); entities that **cannot be
         recognised** (`&foo;`) are then deleted. Deleting them directly would damage
         the text: `it &apos;s` -> `it s`, `&#91;12&#93;` -> `12`, and `&amp;` would
         break `AT&amp;T` into `ATT`, while after restoring they become `it 's`,
         `[12]` and `AT&T`.
       * `unescape=False`: every entity is **deleted** (if that is what you want).

       **Half-width `"` and `'` that are already in the body are not touched**: there
       are 2900+ of them, the vast majority being ordinary quotes in English
       (`the Allied "Thousand Bomber Raids"`), and deleting them would damage the
       English.
    2. The various Unicode spaces (`\\xa0` / ideographic space ...) are normalised to
       a plain space.
    3. **English detokenization** (`detokenize=True`, the default): the English side of
       this corpus is Moses-tokenized - measured: ` ,` on 644 lines, ` .` on 1031
       lines, spaces inside quotes on 220/218 lines, `@-@` on 172 lines:

           no , there 's too much sonic .     ->  no, there's too much sonic.
           fast @-@ acting sugar              ->  fast-acting sugar
           citation &#91;12&#93;  (restored)   ->  citation [12]

       The details and the parts that are deliberately left alone are in
       `detokenize_english`.
    4. **A dash next to CJK becomes the CJK dash `——`** (`dash_scope`, default
       `'cjk'`). Note that the dashes in the Chinese side of this corpus are actually
       written **`- -`** (two hyphens with a space in between) - measured on the
       `wmt_zh_en/5` line as `很 精彩 - - 我只 对 …`; the tight `--` form also occurs:

       * `'cjk'` (the default): only replace when the dash is directly next to CJK,
         eating the spaces on both sides: `CJK--CJK`, `CJK -- CJK` and
         `CJK - - CJK` all become `CJK——CJK`.

       * `'row'`: as long as the line contains CJK, every dash run of that line becomes
         `——` (the spaces on both sides are kept). **Careful**: on a zh-en line this
         also replaces `--` on the English side. Measured over the 26 lines / 115
         occurrences of `--` in this corpus, only one sits next to CJK
         (`贵了一倍 --Sorry`); the rest are English hyphens such as `soft--gluon`,
         English dashes such as `region--owing`, or Markdown separators such as
         `--- **You:**` - so do not switch this on unless it is really needed.
       * `'none'`: leave them alone.

       **A single `-` is not replaced** (a lone hyphen between two CJK words looks
       more like a range or a coordination; measured on 9 lines / 15 occurrences); a
       hyphen with spaces on both sides such as `2 - 7` is left alone too; a cross-line
       Markdown rule (CJK text, `---`, CJK text) is not damaged.
    5. **Strip the spaces between CJK characters**: the Chinese in the WMT branch was
       tokenized and then joined with spaces, and looks like this originally:

            表演的 压轴戏 是 闹剧版 《天鹅湖》 ， 男女 小 人们 身着 粉红色的 芭蕾舞 裙 …

       Only spaces with **CJK on both sides** are eaten, so the spaces between CJK and
       Latin, inside English, and between CJK and digits are all preserved.
    6. Runs of spaces are collapsed into one (steps 1 and 3 tend to leave double spaces
       behind).

    Args:
        text (str): The line to clean.
        unescape (bool): Restore HTML entities instead of deleting all of them.
        detokenize (bool): Apply the English detokenization rules.
        dash_scope (str): Dash rule, one of `DASH_SCOPES`.

    Returns:
        str: The cleaned line.

    Raises:
        ValueError: If `dash_scope` is not one of `DASH_SCOPES`.
    '''
    if not text:
        return text

    if unescape:
        for _ in range(2):
            decoded = html.unescape(text)
            if decoded == text:
                break
            text = decoded
    text = _HTML_ENTITY.sub('', text)
    text = _UNICODE_SPACE.sub(' ', text)
    if detokenize:
        text = detokenize_english(text)

    if dash_scope == 'cjk':
        text = _CJK_DASH.sub('——', text)
    elif dash_scope == 'row':
        if _HAS_CJK.search(text):
            text = _ANY_DASH.sub('——', text)
    elif dash_scope != 'none':
        raise ValueError(f'dash_scope must be one of {DASH_SCOPES}, got {dash_scope!r}.')

    text = _CJK_SPACE.sub('', text)
    text = _SPACE_RUN.sub(' ', text)
    return text


def clean_content(text: str, dataset_type: str = 'none') -> str:
    '''
    Clean one body-text line according to `dataset_type`.

    Args:
        text (str): The line to clean.
        dataset_type (str): 'none' keeps the text as it is, 'tiny_corpus' runs
            `clean_tiny_corpus`.

    Returns:
        str: The cleaned line.
    '''
    if dataset_type == 'tiny_corpus':
        return clean_tiny_corpus(text)
    return text


def as_lang_set(value: Union[None, str, Iterable[str]]) -> Tuple[str, ...]:
    '''
    Normalise a `drop_lang` argument into a tuple of language tags.

    A bare string counts as **one** tag - `drop_lang='en2zh'` drops `en2zh` and not
    the characters `e`, `n`, `2`, `z`, `h` (the classic string-iteration trap) - while
    any other iterable of tags (list / tuple / set / generator) is used as it is.
    `None`, `''` and an empty sequence all mean "drop nothing".

    Args:
        value (Union[None, str, Iterable[str]]): The raw argument.

    Returns:
        Tuple[str, ...]: The tags to drop, `()` when none.

    Raises:
        TypeError: If the argument is not a language tag or an iterable of them.
    '''
    if value is None or value == '':
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        tags = tuple(value)
        if all(isinstance(tag, str) for tag in tags):
            return tags
    raise TypeError(
        f'drop_lang must be a language tag or a sequence of them, got {value!r}.'
    )


class TextFileDataset(MotifDataset):
    '''
    Read text from `root`. Three shapes:

    * `root` is a directory containing `.parquet` - every shard is concatenated in
      file-name order into one data stream (reading the `content` column of every
      row). This is the prev1 sharded layout.
    * `root` is a directory without parquet - the `.txt` files are scanned (recursively
      when `recursive=True`).
    * `root` is a single `.parquet` file - equivalent to a shard set with one shard.

    With a `tokenizer`, `__getitem__` returns the **list of token segments** of that
    file (cut at `seq_len`), which `ChunkedTokenStream` then packs; without one, the
    raw string is returned.

    Args:
        root (str): .txt directory, parquet shard directory, or a single .parquet /
            .txt file.
        recursive (bool): Whether to scan sub-directories recursively (effective for
            both txt and parquet).
        encoding (str): Encoding used to read the .txt files.
        return_path (bool): When True, `__getitem__` returns `(path, content)`; in
            parquet mode the path is the shard file the row lives in (handy for
            tracing a sample back to its source).
        tokenizer (Optional[PackedTokenizer]): Tokenizer; when it is given, `seq_len`
            is required.
        seq_len (Optional[int]): Length of one token segment.
        drop_last (bool): Drop the trailing fragment shorter than `seq_len`.
        shuffle (bool): Shuffle the shard / file order at initialisation using `seed`.
        seed (Optional[int]): Random seed used for shuffling.
        cache_size (int): Number of row groups kept in the lazy-mode cache.
        lazy_mode (bool): True reads row groups on demand; False preloads everything
            at initialisation (with a progress bar).
        column (str): Name of the parquet column holding the body text (prev1 uses
            `content`).
        dataset_type (str): Corpus preprocessing rule, `'none'` (as it is) by default.
            `'tiny_corpus'` = remove the HTML escape entities (such as `&quot;`) and
            strip the spaces between CJK characters (the WMT Chinese was tokenized and
            joined with spaces), see `clean_tiny_corpus`.
        drop_lang (Union[None, str, Iterable[str]]): Language tags (`lang` column) whose rows
            are **dropped while reading** - `('zh2en', 'en2zh')` leaves out the WMT
            translation pairs of TinyCorpus, an empty value (the default) keeps every
            row. A dropped row still occupies its global row number and still gets its
            document boundary token; it only contributes no body tokens. That is what
            makes the filter safe to switch on **mid-run, on a resume**: the cursor in
            the checkpoint is a row number of this dataset, and the row numbering does
            not change. Parquet only (the `.txt` shape has no language tag). The filter is
            part of `state_dict()`, but an *empty* stored value never erases the filter
            the dataset was built with - only a non-empty one has to match, see
            `load_state_dict`.
        lang_column (str): Name of the language-tag column (TinyCorpus calls it
            `lang`); read only when `drop_lang` is set.

    Raises:
        ValueError: If a tokenizer is given without `seq_len`, if `dataset_type` is
            unknown, if `root` is a file that is not a `.parquet`, if `drop_lang` is
            used on a `.txt` corpus, or if the configured `lang_column` is missing.
        TypeError: If `drop_lang` is neither a tag nor a sequence of tags.
        NotADirectoryError: If `root` is neither a directory nor a `.parquet` file.
        FileNotFoundError: If no `.parquet` or `.txt` corpus was found under `root`.
    '''

    def __init__(
        self,
        root: str,
        recursive: bool = False,
        encoding: str = 'utf-8',
        return_path: bool = False,
        tokenizer: Optional[PackedTokenizer] = None,
        seq_len: Optional[int] = None,
        drop_last: bool = True,
        shuffle: bool = False,
        seed: Optional[int] = None,
        cache_size: int = 4,
        lazy_mode: bool = True,
        column: str = 'content',
        dataset_type: str = 'none',
        drop_lang: Union[None, str, Iterable[str]] = None,
        lang_column: str = 'lang',
    ) -> None:
        '''
        Locate the corpus, build the index and preload it when not in lazy mode.

        Args:
            root (str): Corpus directory or a single file.
            recursive (bool): Scan sub-directories recursively.
            encoding (str): Encoding of the .txt files.
            return_path (bool): Return `(path, content)` from `__getitem__`.
            tokenizer (Optional[PackedTokenizer]): Tokenizer used to cut the text into
                token segments.
            seq_len (Optional[int]): Length of one token segment.
            drop_last (bool): Drop a trailing short segment.
            shuffle (bool): Shuffle the shard / file order.
            seed (Optional[int]): Random seed used for shuffling.
            cache_size (int): Row groups kept in the lazy-mode cache.
            lazy_mode (bool): Read row groups on demand instead of preloading.
            column (str): Parquet column holding the body text.
            dataset_type (str): Corpus preprocessing rule.
            drop_lang (Union[None, str, Iterable[str]]): Language tags to drop while
                reading; see the class docstring. `None` / empty keeps every row, a
                bare string counts as one tag.
            lang_column (str): Name of the language-tag column.

        Raises:
            ValueError: If a tokenizer is given without `seq_len`, if `dataset_type` is
                unknown, if `root` is a file that is not a `.parquet`, if `drop_lang` is
                used on a `.txt` corpus, or if `lang_column` is missing from a shard.
            TypeError: If `drop_lang` is neither a tag nor a sequence of tags.
            NotADirectoryError: If `root` is neither a directory nor a `.parquet` file.
            FileNotFoundError: If no corpus was found under `root`.
        '''
        if tokenizer is not None and seq_len is None:
            raise ValueError('seq_len is required when a tokenizer is given.')
        if dataset_type not in DATASET_TYPES:
            raise ValueError(
                f'dataset_type must be one of {DATASET_TYPES}, got {dataset_type!r}.'
            )

        self.root = os.path.abspath(root)
        self.recursive = recursive
        self.encoding = encoding
        self.return_path = return_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.cache_size = max(1, int(cache_size))
        self.lazy_mode = lazy_mode
        self.column = column
        self.dataset_type = dataset_type
        self.drop_lang = as_lang_set(drop_lang)
        self.lang_column = lang_column

        self.files: List[str] = []
        self.parquet_paths: List[str] = []
        self.is_parquet = False
        self.total_rows = 0
        self.data: Optional[List[str]] = None

        self._rg_row_starts: List[int] = []
        self._rg_files: List[int] = []
        self._rg_ids: List[int] = []
        self._file_row_starts: List[int] = []
        self._rg_cache: Dict[int, Any] = {}
        self._rg_order: List[int] = []
        self._pq_file = None
        self._pq_file_idx: int = -1

        if os.path.isfile(self.root):
            if not self.root.lower().endswith('.parquet'):
                raise ValueError(
                    f'{self.root!r} is a file but not a .parquet; pass a directory for a txt corpus.'
                )
            self.parquet_paths = [self.root]
        elif os.path.isdir(self.root):
            self._scan_parquet_files()
            if not self.parquet_paths:
                self._scan_txt_files()
        else:
            raise NotADirectoryError(f'{self.root!r} is neither a directory nor a .parquet file.')

        self.is_parquet = bool(self.parquet_paths)
        if self.drop_lang and not self.is_parquet:
            raise ValueError(
                f'drop_lang={list(self.drop_lang)} needs the parquet layout with a '
                f'{self.lang_column!r} column; a .txt corpus has no language tag.'
            )
        if self.is_parquet:
            self._check_parquet_schema()
            self._init_parquet()
        if not self.parquet_paths and not self.files:
            raise FileNotFoundError(f'no .parquet or .txt corpus was found under {self.root!r}.')

        if not self.lazy_mode:
            if self.is_parquet:
                self._preload_parquet()
            else:
                self._preload_txt_files()

    def _scan_parquet_files(self) -> None:
        '''
        Collect the parquet shards of the directory (sorted by file name, so the order
        is identical across processes and across a resume).
        '''
        paths: List[str] = []
        if self.recursive:
            for dirpath, _, filenames in os.walk(self.root):
                for name in filenames:
                    if name.lower().endswith('.parquet'):
                        paths.append(os.path.join(dirpath, name))
        else:
            for entry in os.scandir(self.root):
                if entry.is_file() and entry.name.lower().endswith('.parquet'):
                    paths.append(entry.path)
        paths.sort()
        if self.shuffle:
            random.Random(self.seed).shuffle(paths)
        self.parquet_paths = paths

    def _scan_txt_files(self) -> None:
        '''
        Collect the `.txt` files of the directory, sorted by file name and optionally
        shuffled with `seed`.
        '''
        if self.recursive:
            for dirpath, _, filenames in os.walk(self.root):
                for fname in filenames:
                    if fname.lower().endswith('.txt'):
                        self.files.append(os.path.join(dirpath, fname))
        else:
            for entry in os.scandir(self.root):
                if entry.is_file() and entry.name.lower().endswith('.txt'):
                    self.files.append(entry.path)
        self.files.sort()
        if self.shuffle:
            random.Random(self.seed).shuffle(self.files)

    def _check_parquet_schema(self) -> None:
        '''
        Fail early: a wrong column name is reported at construction time instead of
        blowing up at row 50,000.

        Raises:
            ValueError: If the body column, or the language-tag column `drop_lang`
                needs, is missing from a shard.
        '''
        for path in self.parquet_paths:
            names = pq.read_schema(path).names
            if self.column not in names:
                raise ValueError(
                    f'{path!r} has no {self.column!r} column (actual columns: {names}).'
                    f'If the body column has a different name, pass column=....'
                )
            if self.drop_lang and self.lang_column not in names:
                raise ValueError(
                    f'{path!r} has no {self.lang_column!r} column (actual columns: '
                    f'{names}), which drop_lang={list(self.drop_lang)} needs to decide '
                    f'what to drop. Pass lang_column=... if it has another name, or '
                    f'leave drop_lang empty.'
                )

    def _init_parquet(self) -> None:
        '''
        Build the global row-group index.

        Only the footer metadata is read (no data), so even thousands of shards are
        fast; `_rg_row_starts` is monotonically increasing, so "global row idx" can be
        bisected straight onto its row group.
        '''
        self._rg_row_starts = []
        self._rg_files = []
        self._rg_ids = []
        self._file_row_starts = [0]
        self._rg_cache.clear()
        self._rg_order.clear()
        self._pq_file = None
        self._pq_file_idx = -1

        total = 0
        for file_idx, path in enumerate(self.parquet_paths):
            meta = pq.read_metadata(path)
            for rg_idx in range(meta.num_row_groups):
                self._rg_row_starts.append(total)
                self._rg_files.append(file_idx)
                self._rg_ids.append(rg_idx)
                total += meta.row_group(rg_idx).num_rows
            self._file_row_starts.append(total)
        self.total_rows = total

    def _preload_parquet(self) -> None:
        '''
        Full preload: read the body text of every shard into memory at once (lazy mode
        does not need this).
        '''
        self.data = []
        with tqdm(total=self.total_rows, desc='Loading Parquet', unit=' rows') as pbar:
            for path in self.parquet_paths:
                reader = pq.ParquetFile(path)
                step = max(1000, reader.metadata.num_rows // 100 or 1000)
                for batch in reader.iter_batches(batch_size=step, columns=[self.column]):
                    texts = ['' if t is None else t for t in batch.column(self.column).to_pylist()]
                    self.data.extend(texts)
                    pbar.update(len(texts))

    def _preload_txt_files(self) -> None:
        '''
        Full preload of the `.txt` files; a file that cannot be read becomes an empty
        string.
        '''
        self.data = []
        with tqdm(total=len(self.files), desc='Loading TXT files', unit='file') as pbar:
            for path in self.files:
                try:
                    with open(path, 'r', encoding=self.encoding) as f:
                        text = f.read()
                except OSError:
                    text = ''
                self.data.append(text)
                pbar.update(1)

    def __len__(self) -> int:
        '''
        Returns:
            int: Number of rows (parquet mode) or of files (txt mode).
        '''
        return self.total_rows if self.is_parquet else len(self.files)

    def count_tokens(self, progress: bool = False) -> int:
        '''
        Walk the whole corpus and count the body tokens it holds.

        The walk takes the same path training takes (`__getitem__`): a `drop_lang` row is
        emptied, `clean_content` cleans the text and `drop_last` drops each row's short
        tail segment - so the number is what the data flow will really consume, not a raw
        character estimate. The document boundary tokens the packer inserts afterwards
        are **not** counted: they are masked out of the loss and are therefore not part
        of a training budget.

        It is a full tokenization pass over the corpus, so it costs roughly one epoch of
        tokenizer time; call it once (the answer does not change) and cache the result
        outside when the corpus is large.

        Args:
            progress (bool): Show a progress bar.

        Returns:
            int: Number of body tokens in the corpus.

        Raises:
            ValueError: If the dataset carries no tokenizer, in which case a token count
                is not defined.
        '''
        if self.tokenizer is None:
            raise ValueError(
                'count_tokens() needs a tokenizer, otherwise there is nothing to count; '
                'build the dataset with tokenizer=...'
            )
        total = 0
        with tqdm(total=len(self), desc='Counting tokens', unit=' rows',
                  disable=not progress) as bar:
            for idx in range(len(self)):
                content = self[idx]
                if self.return_path:
                    _, content = content
                total += sum(len(segment) for segment in content)
                bar.update(1)
        return total

    def _rg_columns(self) -> List[str]:
        '''
        The columns one row group has to be read with.

        Returns:
            List[str]: The body column, plus the language-tag column when the dataset
            drops rows by tag (the tag is needed for every row anyway, so it travels in
            the same read instead of causing a second one).
        '''
        columns = [self.column]
        if self.drop_lang:
            columns.append(self.lang_column)
        return columns

    def _row_group_table(self, rg_idx: int):
        '''
        Lazy mode: fetch the `Table` of one global row group (with an LRU; only the
        current shard handle is kept open).

        Args:
            rg_idx (int): Global row-group index.

        Returns:
            The pyarrow table of that row group.
        '''
        cached = self._rg_cache.get(rg_idx)
        if cached is not None:
            return cached

        file_idx = self._rg_files[rg_idx]
        if self._pq_file_idx != file_idx:
            self._pq_file = pq.ParquetFile(self.parquet_paths[file_idx])
            self._pq_file_idx = file_idx

        table = self._pq_file.read_row_group(self._rg_ids[rg_idx], columns=self._rg_columns())
        if len(self._rg_cache) >= self.cache_size:
            self._rg_cache.pop(self._rg_order.pop(0), None)
        self._rg_cache[rg_idx] = table
        self._rg_order.append(rg_idx)
        return table

    def _read_parquet_row(self, idx: int) -> tuple:
        '''
        Map a global row number to `(shard path, body text)`.

        Args:
            idx (int): Global row number.

        Returns:
            tuple: The shard file the row lives in and its body text.
        '''
        rg_idx = bisect.bisect_right(self._rg_row_starts, idx) - 1
        offset = idx - self._rg_row_starts[rg_idx]
        content = self._row_group_table(rg_idx).column(self.column)[offset].as_py()
        return self.parquet_paths[self._rg_files[rg_idx]], ('' if content is None else content)

    def _read_row(self, idx: int) -> tuple:
        '''
        Read one row as `(path, raw text)`.

        Args:
            idx (int): Row index.

        Returns:
            tuple: The source path and the raw text.

        Raises:
            IOError: If a `.txt` file cannot be read.
        '''
        if not self.lazy_mode:
            return self._path_of(idx), self.data[idx]

        if self.is_parquet:
            return self._read_parquet_row(idx)

        path = self.files[idx]
        try:
            with open(path, 'r', encoding=self.encoding) as f:
                return path, f.read()
        except OSError as e:
            raise IOError(f'failed to read {path!r}: {e}') from e

    def _path_of(self, idx: int) -> str:
        '''
        Which file row `idx` comes from, in full-preload mode.

        Args:
            idx (int): Row index.

        Returns:
            str: The source file path.
        '''
        if not self.is_parquet:
            return self.files[idx]
        file_idx = bisect.bisect_right(self._file_row_starts, idx) - 1
        return self.parquet_paths[file_idx]

    def _lang_at(self, idx: int) -> Any:
        '''
        The `lang` value of global row `idx`.

        The row group it lives in is the one `_read_row` has just fetched (and the LRU
        keeps), so in the default lazy mode reading the tag costs no extra I/O.

        Args:
            idx (int): Global row number.

        Returns:
            Any: The language tag of that row (None when the column holds no value).
        '''
        rg_idx = bisect.bisect_right(self._rg_row_starts, idx) - 1
        offset = idx - self._rg_row_starts[rg_idx]
        return self._row_group_table(rg_idx).column(self.lang_column)[offset].as_py()

    def __getitem__(self, idx: int) -> Union[str, List[List[int]], tuple]:
        '''
        Read one row, clean it and optionally cut it into token segments.

        A row whose language tag is in `drop_lang` is **emptied, not skipped**: it keeps
        its global row number - so the row cursor of a checkpoint stays aligned - and
        contributes no body tokens.

        Args:
            idx (int): Row index.

        Returns:
            Union[str, List[List[int]], tuple]: The (tokenized) content, or
            `(path, content)` when `return_path` is set; empty content (the empty
            string, or an empty segment list when a tokenizer is set) for a dropped
            row.

        Raises:
            IndexError: If `idx` is outside the dataset.
        '''
        if idx < 0 or idx >= len(self):
            raise IndexError(f'Index {idx} out of range (0-{len(self) - 1})')

        path, content = self._read_row(idx)
        if self.drop_lang and self._lang_at(idx) in self.drop_lang:
            content = ''
        content = clean_content(content, self.dataset_type)

        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(
                content, apply_safe_rule=False, add_special_tokens=False
            )
            segments = []
            for i in range(0, len(token_ids), self.seq_len):
                seg = token_ids[i:i + self.seq_len]
                if self.drop_last and len(seg) < self.seq_len:
                    continue
                segments.append(seg)
            content = segments

        return (path, content) if self.return_path else content

    def state_dict(self) -> Dict[str, Any]:
        '''
        Store only the configuration and the shard list, never the corpus itself (a
        corpus can be several GB).

        Returns:
            Dict[str, Any]: The picklable state.
        '''
        state: Dict[str, Any] = {
            'root': self.root,
            'recursive': self.recursive,
            'encoding': self.encoding,
            'return_path': self.return_path,
            'seq_len': self.seq_len,
            'drop_last': self.drop_last,
            'shuffle': self.shuffle,
            'seed': self.seed,
            'cache_size': self.cache_size,
            'lazy_mode': self.lazy_mode,
            'column': self.column,
            'dataset_type': self.dataset_type,
            'drop_lang': list(self.drop_lang),
            'lang_column': self.lang_column,
            'is_parquet': self.is_parquet,
        }
        if self.is_parquet:
            state['parquet_paths'] = list(self.parquet_paths)
            state['total_rows'] = self.total_rows
        else:
            state['files'] = list(self.files)
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        '''
        Restore the configuration and the shard list.

        Rebuilding the index from the shard list is enough because random access is
        stateless and the shard order follows `seed` and is reproducible, so the
        position of a resumed run stays exact. The corpus preprocessing rule and the
        row filter are part of the state, so a resumed run cannot silently switch them.

        Args:
            state (Dict[str, Any]): State produced by `state_dict()`.

        Raises:
            ValueError: If the stored `dataset_type` is unknown, or if the stored
                `drop_lang` contradicts the filter this dataset was built with; the
                corpus preprocessing and the row filter have to match the checkpoint.
        '''
        for key in ('encoding', 'return_path', 'seq_len', 'drop_last',
                    'shuffle', 'seed', 'cache_size', 'lazy_mode', 'column',
                    'dataset_type'):
            if key in state:
                setattr(self, key, state[key])
        if 'drop_lang' in state:
            stored = as_lang_set(state['drop_lang'])
            if stored and stored != self.drop_lang:
                raise ValueError(
                    f'the checkpoint was written while dropping {list(stored)}, but this '
                    f'dataset was built with drop_lang={list(self.drop_lang)}; the row '
                    f'filter has to match the checkpoint (an *empty* stored filter is not '
                    f'a restriction, so it does not erase the filter the dataset was '
                    f'built with and switching the filter on at a resume stays possible).'
                )
        if 'lang_column' in state:
            self.lang_column = state['lang_column']
        if self.dataset_type not in DATASET_TYPES:
            raise ValueError(
                f'the checkpoint carries an unknown dataset_type={self.dataset_type!r}; '
                f'the options are {DATASET_TYPES} (the corpus preprocessing has to match '
                f'the checkpoint).'
            )
        self.cache_size = max(1, int(self.cache_size))

        if state.get('is_parquet', False):
            self.is_parquet = True
            self.parquet_paths = list(state.get('parquet_paths') or [])
            self.files = []
            self._init_parquet()
            if not self.lazy_mode:
                self._preload_parquet()
        else:
            self.is_parquet = False
            self.parquet_paths = []
            if state.get('files') is not None:
                self.files = list(state['files'])
            if not self.lazy_mode:
                self._preload_txt_files()
