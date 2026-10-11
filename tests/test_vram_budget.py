import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from motif.data import MotifDataset
from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.pipeline import PretrainConfig, PretrainPipeline


#: The model part of the 8GB preset (vocab follows the tokenizer; here it is counted as the 40960 of this library's tokenizer).
EIGHT_GB_MODEL = dict(
    model_dim=320, num_heads=4, num_kv_heads=2, head_k_dim=64,
    block=('gdn', 'chord'), block_n=8,
)
EIGHT_GB_VOCAB = 40960

#: The pretraining part of the 8GB preset.
EIGHT_GB_PRETRAIN = dict(
    base_context=512, target_context=2048, global_batch_tokens=6144,
    step_mode='min', learning_rate=6e-4, weight_decay=0.05, warmup_steps=200,
    grad_clip_norm=1.0, amp=True,
)

#: The discrete-GPU budget (= the default of pretrain_prev1.py's --vram-budget-gb).
EIGHT_GB_VRAM_BUDGET_GIB = 6.5

#: Peaks measured on this machine (RTX 5070 Laptop, 7.96 GiB).
MEASURED_SEQ512_ALLOC_GIB = 4.96
MEASURED_SEQ512_RESERVED_GIB = 6.09
MEASURED_SEQ2048_ALLOC_GIB = 3.86
MEASURED_SEQ2048_RESERVED_GIB = 4.64

CUDA = torch.cuda.is_available()
TOTAL_GIB = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3) if CUDA else 0.0
FREE_GIB = (torch.cuda.mem_get_info()[0] / 1024 ** 3) if CUDA else 0.0

#: The measurement needs ~6.1 GiB reserved; when free memory is below this, do not grab the GPU yet (another process may be running).
NEED_FREE_GIB = 5.5


def _skip(reason: str):
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        print(f'[skip] {reason}')
        return None


def build_8gb_configs(vocab_size: int = EIGHT_GB_VOCAB, **overrides):
    '''Turn the 8GB preset into `(MotifChordConfig, PretrainConfig)` (a local reproduction can just copy these two).'''
    model_kwargs = {**EIGHT_GB_MODEL, 'vocab_size': vocab_size}
    model_kwargs.update(overrides.pop('model', {}))
    pretrain_kwargs = {**EIGHT_GB_PRETRAIN}
    pretrain_kwargs.update(overrides.pop('pretrain', {}))
    return MotifChordConfig(**model_kwargs), PretrainConfig(**pretrain_kwargs)


def state_gib(params: int) -> float:
    '''
    fp32 training state is about params × 16 bytes: master weights 4 + gradients 4 + AdamW's two moments 8.

    `amp=True` only affects activations (bf16 autocast); the optimizer state is still fp32.
    '''
    return params * 16 / 1024 ** 3


class ListDataset(MotifDataset):
    '''Synthetic token sequences: this test only cares about VRAM and needs no real corpus / tokenizer.'''

    def __init__(self, n, length, vocab=EIGHT_GB_VOCAB):
        self.records = [torch.randint(20, vocab, (length,)).tolist() for _ in range(n)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


class NoSave(PretrainPipeline):
    '''Skip the checkpoint write to disk: this test only measures VRAM and avoids dropping a 0.5 GiB weight file.'''

    def save_checkpoint(self, path: str) -> None:
        return None


def make_pipeline(model_config, pretrain_config, device='cuda'):
    torch.manual_seed(0)
    model = MotifChordModel(model_config).to(device)
    pretrain_config.ckpt_dir = os.path.join(os.environ.get('TEMP', '.'), 'unused_vram_ckpt')
    pretrain_config.probe_every_steps = 0
    pretrain_config.use_progress = False
    pretrain_config.eos_token_id = 9
    return NoSave(model, None, pretrain_config, device=device, seed=0)


def test_8gb_preset_state_fits_budget():
    model_config, pretrain_config = build_8gb_configs()
    params = sum(p.numel() for p in MotifChordModel(model_config).parameters())

    assert params < 40e6, f'the 8GB preset should not exceed 40M parameters, got {params/1e6:.1f} M'
    state = state_gib(params)
    assert state < EIGHT_GB_VRAM_BUDGET_GIB / 2, (
        f'training state alone is {state:.2f} GiB, and taking more than half of the '
        f'{EIGHT_GB_VRAM_BUDGET_GIB} GiB budget leaves no room for activations')

    dashboard = sum(p.numel() for p in MotifChordModel(
        MotifChordConfig(vocab_size=40960, block_n=6)).parameters())
    assert state_gib(dashboard) > 2.5, (
        f'the block_n=6 default configuration should be clearly more expensive (measured {state_gib(dashboard):.2f} GiB)')


def test_8gb_preset_matches_documented_numbers():
    model_config, pretrain_config = build_8gb_configs()
    assert model_config.model_dim == 320 and model_config.block_n == 8
    assert model_config.block == ('gdn', 'chord'), 'only two decoders (gdn+chord) per layer is cheap enough'
    assert pretrain_config.base_context == 512
    assert pretrain_config.target_context == 2048
    assert pretrain_config.global_batch_tokens == 6144, 'the token count per forward determines peak VRAM'
    assert pretrain_config.amp is True, 'bf16 autocast is the precondition for this configuration fitting into 8GB'
    assert EIGHT_GB_VRAM_BUDGET_GIB == 6.5


def test_8gb_preset_runs_within_budget_on_cuda():
    if not CUDA:
        return _skip('no CUDA, skipping the VRAM measurement')
    if FREE_GIB < NEED_FREE_GIB:
        return _skip(f'free VRAM {FREE_GIB:.2f} GiB < {NEED_FREE_GIB} GiB, not grabbing the GPU for now')

    model_config, pretrain_config = build_8gb_configs()
    pipe = make_pipeline(model_config, pretrain_config)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        pipe.train(ListDataset(12, length=1024), num_epochs=1, steps_per_epoch=2)
        step = pipe.state.global_step
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    finally:
        del pipe
        torch.cuda.empty_cache()

    assert step == 2, 'the measurement should really complete 2 steps'
    assert alloc <= 6.0, f'peak allocated {alloc:.2f} GiB exceeds the usable headroom of an 8GB card'
    assert reserved <= 7.0, f'peak reserved {reserved:.2f} GiB is hugging the 8GB cap'
    assert reserved >= MEASURED_SEQ512_RESERVED_GIB - 1.0, (
        f'peak reserved is only {reserved:.2f} GiB, far below the recorded {MEASURED_SEQ512_RESERVED_GIB} GiB '
        f'-- the configuration may have been shrunk, which would make this test meaningless')
    print(f'[vram] seq512 stage: peak allocated {alloc:.2f} GiB / reserved {reserved:.2f} GiB')


def test_8gb_preset_long_context_stage_fits():
    '''The seq=2048 (target_context) stage is where OOM actually happens.'''
    if not CUDA:
        return _skip('no CUDA, skipping the VRAM measurement')
    if FREE_GIB < NEED_FREE_GIB:
        return _skip(f'free VRAM {FREE_GIB:.2f} GiB < {NEED_FREE_GIB} GiB, not grabbing the GPU for now')

    model_config, pretrain_config = build_8gb_configs(pretrain={'base_context': 2048,
                                                               'target_context': 4096})
    pipe = make_pipeline(model_config, pretrain_config)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        pipe.setup()
        runner = next(pipe.iterate_epochs(ListDataset(4, length=4096)))
        iterator = iter(runner)
        for _ in range(2):
            pipe._optimizer_step([next(iterator)], 1.0, True)
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
    finally:
        del pipe
        torch.cuda.empty_cache()

    assert alloc <= 6.0, f'seq 2048 stage peak allocated {alloc:.2f} GiB is too high'
    assert reserved <= 7.0, f'seq 2048 stage peak reserved {reserved:.2f} GiB is hugging the 8GB cap'
    print(f'[vram] seq2048 stage: peak allocated {alloc:.2f} GiB / reserved {reserved:.2f} GiB')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all vram budget tests passed')
