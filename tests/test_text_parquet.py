import os
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from motif.data import ChunkedTokenStream, TextFileDataset, clean_tiny_corpus
from motif.pipeline import PretrainConfig, PretrainPipeline

from support import make_model, make_tokenizer


def write_shards(directory: str, shards: int, rows: int, row_group_size: int = 7,
                 column: str = 'content') -> str:
    '''Write a batch of shard parquet files, each split into several row groups (so that cross-group / cross-shard indexing gets exercised).'''
    os.makedirs(directory, exist_ok=True)
    for shard in range(shards):
        table = pa.table({
            column: [f'shard{shard:02d}-row{row:03d}' for row in range(rows)],
            'tag': [shard] * rows,
        })
        pq.write_table(
            table,
            os.path.join(directory, f'part-{shard:03d}.parquet'),
            row_group_size=row_group_size,
        )
    return directory


def expected_rows(shards: int, rows: int):
    for shard in range(shards):
        for row in range(rows):
            yield f'shard{shard:02d}-row{row:03d}'


def test_parquet_folder_is_one_logical_stream():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=5, rows=11)
        for lazy in (True, False):
            dataset = TextFileDataset(folder, lazy_mode=lazy)
            assert dataset.is_parquet
            assert len(dataset) == 5 * 11
            assert list(dataset) == list(expected_rows(5, 11)), f'lazy_mode={lazy} concatenation order is wrong'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_parquet_lazy_and_full_agree():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=4, rows=13, row_group_size=5)
        lazy = TextFileDataset(folder, lazy_mode=True, cache_size=2)
        full = TextFileDataset(folder, lazy_mode=False)
        assert [lazy[i] for i in range(len(lazy))] == [full[i] for i in range(len(full))]
        assert lazy.parquet_paths == full.parquet_paths
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_parquet_recursive_scan():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        write_shards(os.path.join(base, 'shards', 'a'), shards=2, rows=5)
        write_shards(os.path.join(base, 'shards', 'b'), shards=3, rows=5)

        try:
            TextFileDataset(os.path.join(base, 'shards'), recursive=False)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('a non-recursive scan must not find parquet under subdirectories')

        deep = TextFileDataset(os.path.join(base, 'shards'), recursive=True)
        assert len(deep) == 5 * 5
        assert len(deep.parquet_paths) == 5
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_parquet_return_path_points_at_the_shard():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=3, rows=4)
        dataset = TextFileDataset(folder, return_path=True)

        seen = set()
        rows = list(expected_rows(3, 4))
        for i, want in enumerate(rows):
            path, content = dataset[i]
            assert os.path.isfile(path)
            assert content == want, f'row {i} content mismatch'
            assert os.path.basename(path) == f'part-{i // 4:03d}.parquet'
            seen.add(os.path.basename(path))

        assert seen == {'part-000.parquet', 'part-001.parquet', 'part-002.parquet'}
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_single_parquet_file_still_works():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'one'), shards=1, rows=9, row_group_size=4)
        path = os.path.join(folder, 'part-000.parquet')
        dataset = TextFileDataset(path)
        assert dataset.is_parquet
        assert len(dataset) == 9
        assert list(dataset) == list(expected_rows(1, 9))
        assert dataset.parquet_paths == [path]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_parquet_custom_column_and_missing_column():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=2, rows=3, column='text')
        dataset = TextFileDataset(folder, column='text')
        assert len(dataset) == 6

        try:
            TextFileDataset(folder)
        except ValueError as exc:
            assert "'content'" in str(exc) and "'text'" in str(exc)
        else:
            raise AssertionError('a mismatched column name was not reported')
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_txt_directory_still_works():
    base = tempfile.mkdtemp(prefix='motif_txt_')
    try:
        root = os.path.join(base, 'txt')
        os.makedirs(root)
        for i in range(3):
            with open(os.path.join(root, f'{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(f'doc {i}')
        dataset = TextFileDataset(root)
        assert not dataset.is_parquet
        assert list(dataset) == ['doc 0', 'doc 1', 'doc 2']
    finally:
        shutil.rmtree(base, ignore_errors=True)


#: The wmt_zh_en/2 row from the screenshot (the Chinese side is tokenized and rejoined with spaces, the English side was Moses-tokenized).
WMT_ROW = ('表演的 压轴戏 是 闹剧版 《天鹅湖》 ， 男女 小 人们 身着 粉红色的 芭蕾舞 裙 '
           '扮演 小天鹅 。 the centerpiece of the show is a farcical rendition of Swan Lake '
           'in which male and female performers dance in pink tutus and imitate swans .')
WMT_CLEAN = ('表演的压轴戏是闹剧版《天鹅湖》，男女小人们身着粉红色的芭蕾舞裙扮演小天鹅。 '
             'the centerpiece of the show is a farcical rendition of Swan Lake in which '
             'male and female performers dance in pink tutus and imitate swans.')


def test_clean_tiny_corpus_removes_cjk_spaces():
    assert clean_tiny_corpus(WMT_ROW) == WMT_CLEAN
    assert clean_tiny_corpus('中文 English 混排 文本') == '中文 English 混排文本'
    assert clean_tiny_corpus('a  b   c') == 'a b c'
    assert clean_tiny_corpus('第 3 章 内容') == '第 3 章内容'
    assert clean_tiny_corpus('中文\u3000全角空格') == '中文全角空格'
    assert clean_tiny_corpus('中文\u00a0nbsp') == '中文 nbsp'
    assert clean_tiny_corpus('结束 ， 下一句') == '结束，下一句'
    assert clean_tiny_corpus(WMT_CLEAN) == WMT_CLEAN


def test_clean_tiny_corpus_handles_html_entities():
    assert clean_tiny_corpus('他说&quot;你好&quot;') == '他说"你好"'
    assert clean_tiny_corpus('no , there &apos;s too much', detokenize=False) == \
        "no , there 's too much"
    assert clean_tiny_corpus('&lt;div&gt;') == '<div>'
    assert clean_tiny_corpus('citation &#91;12&#93;') == 'citation [12]'
    assert clean_tiny_corpus('it&#39;s') == "it's"
    assert clean_tiny_corpus('&#x27;quoted&#x27;') == "'quoted'"
    assert clean_tiny_corpus('AT&amp;T 与 R&amp;D') == 'AT&T 与 R&D'
    assert clean_tiny_corpus('&#38; 与 &#x26;') == '& 与 &'
    assert clean_tiny_corpus('x&nbsp;y') == 'x y'
    assert clean_tiny_corpus('a &amp;quot; b') == 'a " b'
    assert clean_tiny_corpus('a &foo; b') == 'a b'
    assert clean_tiny_corpus('A &foo; B') == 'A B'
    assert clean_tiny_corpus('AT&T 与 R&D') == 'AT&T 与 R&D'
    assert clean_tiny_corpus('the Allied "Thousand Bomber Raids"') == \
        'the Allied "Thousand Bomber Raids"'
    assert clean_tiny_corpus("don't 删除") == "don't 删除"


def test_clean_tiny_corpus_detokenizes_english():
    assert clean_tiny_corpus('there &apos;s too much') == "there's too much"
    assert clean_tiny_corpus('it &apos;s highly unlikely') == "it's highly unlikely"
    assert clean_tiny_corpus('no , there &apos;s too much sonic .') == \
        "no, there's too much sonic."
    assert clean_tiny_corpus("do n't go") == "don't go"
    assert clean_tiny_corpus("I 'm here") == "I'm here"
    assert clean_tiny_corpus("they 're , we 've , he 'll , she 'd , can 't") == \
        "they're, we've, he'll, she'd, can't"
    assert clean_tiny_corpus('no , sir . really ? yes ; ok : 50 %') == \
        'no, sir. really? yes; ok: 50%'
    assert clean_tiny_corpus('( 1 ) [ 2 ] { 3 }') == '(1) [2] {3}'
    assert clean_tiny_corpus('said " hello " to me') == 'said " hello " to me'
    assert clean_tiny_corpus('fast @-@ acting sugar') == 'fast-acting sugar'
    assert clean_tiny_corpus('peoples 2 - 7 3 .') == 'peoples 2 - 7 3.'
    assert clean_tiny_corpus('中文 ， 结束 。 English !') == '中文，结束。 English!'
    assert clean_tiny_corpus("no , there 's too much .", detokenize=False) == \
        "no , there 's too much ."
    assert clean_tiny_corpus("no, there's too much sonic.") == "no, there's too much sonic."


def test_clean_tiny_corpus_chinese_dash():
    assert clean_tiny_corpus('中文--中文') == '中文——中文'
    assert clean_tiny_corpus('中文 -- 中文') == '中文——中文'
    assert clean_tiny_corpus('中文 --- 中文') == '中文——中文'
    assert clean_tiny_corpus('他说 --好吧') == '他说——好吧'
    assert clean_tiny_corpus('--中文开头') == '——中文开头'
    assert clean_tiny_corpus('中文结尾--') == '中文结尾——'
    assert clean_tiny_corpus('很 精彩 - - 我只 对 她的 技巧') == '很精彩——我只对她的技巧'
    assert clean_tiny_corpus('精彩 -  -  我只') == '精彩——我只'
    assert clean_tiny_corpus('精彩 - - - 我只') == '精彩——我只'
    assert clean_tiny_corpus('- - 中文开头') == '——中文开头'
    assert clean_tiny_corpus('soft--gluon resummation') == 'soft--gluon resummation'
    assert clean_tiny_corpus('person -- Dewi is scheduled') == 'person -- Dewi is scheduled'
    assert clean_tiny_corpus('columns (4)--(7)') == 'columns (4)--(7)'
    assert clean_tiny_corpus('中文\n---\n英文') == '中文\n---\n英文'
    assert clean_tiny_corpus('中国 - 美国') == '中国 - 美国'
    assert clean_tiny_corpus('peoples 2 - 7 3') == 'peoples 2 - 7 3'
    assert clean_tiny_corpus('中文 person -- Dewi', dash_scope='row') == '中文 person —— Dewi'
    assert clean_tiny_corpus('person -- Dewi', dash_scope='row') == 'person -- Dewi'
    assert clean_tiny_corpus('中国 2 - 7 美国', dash_scope='row') == '中国 2 - 7 美国'
    assert clean_tiny_corpus('中文 -- 中文', dash_scope='none') == '中文 -- 中文'
    assert clean_tiny_corpus('精彩 - - 我只', dash_scope='none') == '精彩 - - 我只'
    try:
        clean_tiny_corpus('中文 -- 中文', dash_scope='nope')
    except ValueError as exc:
        assert 'dash_scope' in str(exc)
    else:
        raise AssertionError('an illegal dash_scope must raise')


def test_clean_tiny_corpus_can_delete_all_entities():
    '''`unescape=False`: every entity is deleted unconditionally (the "delete things like &quot;" you originally meant).'''
    assert clean_tiny_corpus('他说&quot;你好&quot;', unescape=False) == '他说你好'
    assert clean_tiny_corpus('no , there &apos;s', unescape=False, detokenize=False) == \
        'no , there s'
    assert clean_tiny_corpus('no , there &apos;s', unescape=False) == 'no, there s'
    assert clean_tiny_corpus('citation &#91;12&#93;', unescape=False) == 'citation 12'
    assert clean_tiny_corpus('AT&amp;T', unescape=False) == 'ATT'


def _write_clean_shard(directory: str, rows) -> str:
    os.makedirs(directory, exist_ok=True)
    table = pa.table({'content': rows, 'lang': ['zh2en'] * len(rows)})
    pq.write_table(table, os.path.join(directory, 'part-000.parquet'))
    return directory


def test_dataset_type_defaults_to_none():
    base = tempfile.mkdtemp(prefix='motif_type_')
    try:
        folder = _write_clean_shard(os.path.join(base, 'shards'), [WMT_ROW])
        dataset = TextFileDataset(folder)
        assert dataset.dataset_type == 'none'
        assert list(dataset) == [WMT_ROW], 'the default must not touch the body text'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_dataset_type_tiny_corpus_cleans_and_shrinks_tokens():
    base = tempfile.mkdtemp(prefix='motif_type_')
    try:
        folder = _write_clean_shard(os.path.join(base, 'shards'), [WMT_ROW, WMT_ROW])

        raw = TextFileDataset(folder)
        clean = TextFileDataset(folder, dataset_type='tiny_corpus')
        assert list(clean) == [WMT_CLEAN, WMT_CLEAN]
        assert list(clean) != list(raw)

        tokenizer = make_tokenizer()
        raw_tok = TextFileDataset(folder, tokenizer=tokenizer, seq_len=8, drop_last=False)
        clean_tok = TextFileDataset(folder, dataset_type='tiny_corpus',
                                    tokenizer=tokenizer, seq_len=8, drop_last=False)
        raw_tokens = sum(len(seg) for seg in raw_tok[0])
        clean_tokens = sum(len(seg) for seg in clean_tok[0])
        assert clean_tokens < raw_tokens, \
            f'cleaning should yield fewer tokens: {clean_tokens} vs {raw_tokens}'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_dataset_type_validation_and_state_roundtrip():
    base = tempfile.mkdtemp(prefix='motif_type_')
    try:
        folder = _write_clean_shard(os.path.join(base, 'shards'), [WMT_ROW])
        try:
            TextFileDataset(folder, dataset_type='nope')
        except ValueError as exc:
            assert 'tiny_corpus' in str(exc)
        else:
            raise AssertionError('an unknown dataset_type must raise')

        dataset = TextFileDataset(folder, dataset_type='tiny_corpus')
        state = dataset.state_dict()
        assert state['dataset_type'] == 'tiny_corpus'

        restored = TextFileDataset(folder)
        restored.load_state_dict(state)
        assert restored.dataset_type == 'tiny_corpus'
        assert list(restored) == [WMT_CLEAN], 'the restored dataset must still clean with the same recipe'

        bad = TextFileDataset(folder)
        bad_state = dict(state, dataset_type='whatever')
        try:
            bad.load_state_dict(bad_state)
        except ValueError as exc:
            assert 'dataset_type' in str(exc)
        else:
            raise AssertionError('an archive with an unknown recipe must raise')
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_parquet_state_roundtrip():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=3, rows=6)
        dataset = TextFileDataset(folder, tokenizer=make_tokenizer(), seq_len=8)
        state = dataset.state_dict()
        assert state['is_parquet'] and len(state['parquet_paths']) == 3

        restored = TextFileDataset(folder)
        restored.load_state_dict(state)
        assert restored.state_dict() == state
        assert restored.parquet_paths == dataset.parquet_paths
        assert len(restored) == len(dataset)

        full = TextFileDataset(folder, lazy_mode=False, tokenizer=make_tokenizer(), seq_len=8)
        restored_full = TextFileDataset(folder, tokenizer=make_tokenizer(), seq_len=8)
        restored_full.load_state_dict(full.state_dict())
        assert [restored_full[i] for i in range(len(full))] == [full[i] for i in range(len(full))]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_chunked_stream_resume_over_parquet_shards():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=4, rows=6)
        tokenizer = make_tokenizer()

        def make():
            dataset = TextFileDataset(folder, tokenizer=tokenizer, seq_len=12, drop_last=False)
            return ChunkedTokenStream(dataset, chunk_len=2 * 13, batch_size=2,
                                      seq_len=12, eos_token_id=2)

        full = list(make())
        assert len(full) >= 6, 'the sharded corpus is too small to exercise a mid-stream checkpoint'

        stream = make()
        iterator = iter(stream)
        partial = [next(iterator) for _ in range(4)]
        state = stream.state_dict()
        del iterator

        resumed = make()
        resumed.load_state_dict(state)
        rest = list(resumed)

        assert len(partial) + len(rest) == len(full)
        for got, want in zip(partial + rest, full):
            assert torch.equal(got[0], want[0]), 'cross-shard resume input_ids are misaligned'
            assert torch.equal(got[1], want[1]), 'cross-shard resume labels are misaligned'
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_pretrain_pipeline_over_parquet_shards_resumes():
    '''Sharded corpora must resume end to end as well: the batches seen before and after the break match uninterrupted training.'''
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_shards(os.path.join(base, 'shards'), shards=4, rows=8)
        tokenizer = make_tokenizer()

        from motif.pipeline import Callback

        class StopAfter(Callback):
            def __init__(self, n):
                self.n = n

            def on_step_end(self, pipeline, step_metrics):
                if pipeline.state.global_step >= self.n:
                    self.stop_requested = True

        class Recording(PretrainPipeline):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seen = []

            def compute_loss(self, batch):
                stage, inputs, labels = batch
                self.seen.append((stage.name, inputs.clone(), labels.clone()))
                return super().compute_loss(batch)

        def make_dataset():
            return TextFileDataset(folder, tokenizer=tokenizer, seq_len=16, drop_last=False)

        def cfg(ckpt):
            return PretrainConfig(
                base_context=16, target_context=32, global_batch_tokens=32, step_mode='min',
                learning_rate=1e-3, warmup_steps=1, ckpt_dir=ckpt, save_every_steps=0,
                probe_every_steps=0, use_progress=False, amp=False)

        reference = Recording(make_model(vocab_size=tokenizer.vocab_size), tokenizer, cfg(os.path.join(base, 'ck_ref')),
                              callbacks=[StopAfter(5)], seed=0, device='cpu')
        reference.train(make_dataset(), num_epochs=1)
        assert len(reference.seen) == 5

        ckpt_dir = os.path.join(base, 'ck')
        first = Recording(make_model(vocab_size=tokenizer.vocab_size), tokenizer, cfg(ckpt_dir),
                          callbacks=[StopAfter(3)], seed=0, device='cpu')
        first.train(make_dataset(), num_epochs=1)
        assert len(first.seen) == 3

        second = Recording(make_model(vocab_size=tokenizer.vocab_size), tokenizer, cfg(ckpt_dir),
                           callbacks=[StopAfter(5)], seed=0, device='cpu')
        second.train(make_dataset(), num_epochs=2)
        assert len(second.seen) == 2

        for (gs, gx, gy), (ws, wx, wy) in zip(first.seen + second.seen, reference.seen):
            assert gs == ws
            assert torch.equal(gx, wx), 'after resuming a sharded corpus the input_ids differ from uninterrupted training'
            assert torch.equal(gy, wy), 'after resuming a sharded corpus the labels differ from uninterrupted training'
    finally:
        shutil.rmtree(base, ignore_errors=True)


LANG_PATTERN = ('zh2en', 'en2zh', None, 'en')


def lang_of(row: int):
    '''The `lang` of row `row`: one cycle every 4 rows, consistent across shards, so "which rows get dropped" is computable.'''
    return LANG_PATTERN[row % len(LANG_PATTERN)]


def translation_rows(rows: int = 21):
    '''The row numbers of the `zh2en` / `en2zh` (bidirectional translation) rows within one shard.'''
    return {row for row in range(rows) if lang_of(row) in ('zh2en', 'en2zh')}


def write_lang_shards(directory: str, shards: int = 3, rows: int = 21,
                      row_group_size: int = 7, skip=()) -> str:
    '''
    Write a batch of parquet shards carrying a `lang` column.

    The row numbers in `skip` are **not written into the parquet**; they are used to build a control corpus in
    which those two directions were never present.
    '''
    os.makedirs(directory, exist_ok=True)
    skip = set(skip)
    for shard in range(shards):
        kept = [row for row in range(rows) if row not in skip]
        table = pa.table({
            'content': [f'shard{shard:02d}-row{row:03d}' for row in kept],
            'lang': [lang_of(row) for row in kept],
        })
        pq.write_table(
            table,
            os.path.join(directory, f'part-{shard:03d}.parquet'),
            row_group_size=row_group_size,
        )
    return directory


def test_drop_lang_empties_rows_without_moving_the_row_index():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_lang_shards(os.path.join(base, 'shards'), shards=3, rows=21)
        plain = TextFileDataset(folder)
        filtered = TextFileDataset(folder, drop_lang=('zh2en', 'en2zh'))

        assert len(filtered) == len(plain) == 3 * 21
        assert filtered.parquet_paths == plain.parquet_paths
        assert sum(1 for i in range(len(filtered)) if filtered[i] == '') == 3 * len(translation_rows())

        for i in range(len(plain)):
            shard, row = divmod(i, 21)
            if lang_of(row) in ('zh2en', 'en2zh'):
                assert filtered[i] == '', f'row {i} with lang={lang_of(row)} was not dropped'
            else:
                assert filtered[i] == plain[i] == f'shard{shard:02d}-row{row:03d}', f'row {i} was damaged by mistake'

        trimmed = TextFileDataset(write_lang_shards(
            os.path.join(base, 'trimmed'), shards=3, rows=21, skip=translation_rows()))
        assert len(trimmed) == 3 * (21 - len(translation_rows()))
        assert [v for v in (filtered[i] for i in range(len(filtered))) if v] == list(trimmed)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_drop_lang_accepts_one_tag_as_a_bare_string():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_lang_shards(os.path.join(base, 'shards'), shards=2, rows=9)
        one_way = TextFileDataset(folder, drop_lang='en2zh')
        assert one_way.drop_lang == ('en2zh',)

        rows = [one_way[i] for i in range(len(one_way))]
        assert rows[1] == rows[5] == rows[10] == '', 'rows with lang=en2zh should be dropped'
        assert rows[0] != '' and rows[2] != '' and rows[3] != '', 'zh2en / None / en should all be kept'
        assert rows[9] != '', 'when only en2zh is dropped, the other direction (zh2en) must be kept as usual'

        assert TextFileDataset(folder, drop_lang={'en2zh'}).drop_lang == ('en2zh',)
        assert TextFileDataset(folder, drop_lang=['zh2en', 'en2zh']).drop_lang == ('zh2en', 'en2zh')
        for empty in (None, '', ()):
            assert TextFileDataset(folder, drop_lang=empty).drop_lang == ()
            assert TextFileDataset(folder, drop_lang=empty)[1] != ''

        for bad in (3, ['en', 3], b'en2zh'):
            try:
                TextFileDataset(folder, drop_lang=bad)
            except TypeError:
                pass
            else:
                raise AssertionError(f'drop_lang={bad!r} should raise TypeError')
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_drop_lang_needs_a_lang_column_and_a_parquet_corpus():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        tagged = write_shards(os.path.join(base, 'tagged'), shards=2, rows=6)
        TextFileDataset(tagged, drop_lang=('en',), lang_column='tag')

        try:
            TextFileDataset(tagged, drop_lang=('en',))
        except ValueError as exc:
            assert 'lang' in str(exc)
        else:
            raise AssertionError('a missing lang column should raise instead of silently dropping nothing')

        txt = os.path.join(base, 'txt')
        os.makedirs(txt, exist_ok=True)
        with open(os.path.join(txt, 'a.txt'), 'w', encoding='utf-8') as handle:
            handle.write('hello 你好\n')
        try:
            TextFileDataset(txt, drop_lang=('en',))
        except ValueError as exc:
            assert 'parquet' in str(exc)
        else:
            raise AssertionError('.txt corpora carry no language tag, so drop_lang should raise')
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_drop_lang_travels_in_the_state():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        folder = write_lang_shards(os.path.join(base, 'shards'), shards=2, rows=12)
        dataset = TextFileDataset(folder, drop_lang=('zh2en', 'en2zh'))
        state = dataset.state_dict()
        assert state['drop_lang'] == ['zh2en', 'en2zh'] and state['lang_column'] == 'lang'

        restored = TextFileDataset(folder, drop_lang=('zh2en', 'en2zh'))
        restored.load_state_dict(state)
        assert restored.state_dict() == state
        assert [restored[i] for i in range(len(restored))] == [dataset[i] for i in range(len(dataset))]

        fresh = TextFileDataset(folder, drop_lang=('zh2en', 'en2zh'))
        fresh.load_state_dict(TextFileDataset(folder).state_dict())
        assert fresh.drop_lang == ('zh2en', 'en2zh')

        try:
            TextFileDataset(folder).load_state_dict(state)
        except ValueError as exc:
            assert 'drop_lang' in str(exc)
        else:
            raise AssertionError('the filter in the checkpoint does not match the one given at construction; this should raise')
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_drop_lang_kept_rows_tokenize_exactly_as_without_them():
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        tokenizer = make_tokenizer()
        shards, rows, seq_len = 3, 21, 12
        folder = write_lang_shards(os.path.join(base, 'shards'), shards=shards, rows=rows)
        trimmed = write_lang_shards(os.path.join(base, 'trimmed'), shards=shards, rows=rows,
                                    skip=translation_rows(rows))

        filtered = TextFileDataset(folder, tokenizer=tokenizer, seq_len=seq_len, drop_last=False,
                                   drop_lang=('zh2en', 'en2zh'))
        reference = TextFileDataset(trimmed, tokenizer=tokenizer, seq_len=seq_len, drop_last=False)

        for i in range(21):
            if lang_of(i) in ('zh2en', 'en2zh'):
                assert filtered[i] == [], 'a dropped row should yield an empty segment list when tokenized'
        assert [filtered[i] for i in range(len(filtered)) if filtered[i]] == \
               [reference[i] for i in range(len(reference))]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_switching_drop_lang_on_a_resume_keeps_the_same_cursor():
    '''
    The reason this switch exists: the only data cursor in the checkpoint is "which row we have read up to", and
    dropping content does not move row numbers, so turning it on at resume takes effect in place, without
    replaying or skipping a single row.
    '''
    base = tempfile.mkdtemp(prefix='motif_pq_')
    try:
        tokenizer = make_tokenizer()
        shards, rows, seq_len = 3, 21, 12
        folder = write_lang_shards(os.path.join(base, 'shards'), shards=shards, rows=rows)

        class Recording(TextFileDataset):
            '''Records which rows were actually read (replays / skips would show up here).'''

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.read = []

            def __getitem__(self, idx):
                self.read.append(idx)
                return super().__getitem__(idx)

        def stream(dataset):
            return ChunkedTokenStream(dataset, chunk_len=2 * (seq_len + 1), batch_size=2,
                                      seq_len=seq_len, eos_token_id=2)

        before = stream(TextFileDataset(folder, tokenizer=tokenizer, seq_len=seq_len, drop_last=False))
        iterator = iter(before)
        for _ in range(3):
            next(iterator)
        state = before.state_dict()
        del iterator
        offset = state['upstream_offset']
        assert offset > 0 and offset < shards * rows, 'this corpus did not cross the checkpoint'

        dataset = Recording(folder, tokenizer=tokenizer, seq_len=seq_len, drop_last=False,
                            drop_lang=('zh2en', 'en2zh'))
        after = stream(dataset)
        after.load_state_dict(state)
        assert after.state_dict()['upstream_offset'] == offset, 'the cursor was moved after resume'

        consumed = sum(1 for _ in after)
        assert consumed > 0, 'after dropping the translations it must still be possible to train'

        read = list(dataset.read)
        assert read == list(range(offset, offset + len(read))), \
            f'after resume it must keep reading from row {offset}: neither going back nor stepping over'
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all parquet tests passed')
