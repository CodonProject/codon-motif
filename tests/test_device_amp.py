import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from motif.pipeline import resolve_device


CUDA = torch.cuda.is_available()


def test_explicit_device_wins():
    model = nn.Linear(4, 4)
    assert resolve_device(model, 'cpu').type == 'cpu'


def test_cpu_model_does_not_silently_mean_cpu_training():
    '''With a model on the CPU, auto must still choose CUDA on a machine that has CUDA.'''
    model = nn.Linear(4, 4)
    resolved = resolve_device(model, None)
    assert resolved.type == ('cuda' if CUDA else 'cpu')
    assert resolve_device(model, 'auto').type == resolved.type


def test_model_already_on_device_is_followed():
    if not CUDA:
        return
    model = nn.Linear(4, 4).cuda()
    assert resolve_device(model, None).type == 'cuda'
    assert resolve_device(model, None) == torch.device('cuda:0')


def test_declared_device_attribute_is_honoured():
    if not CUDA:
        return

    class Fake:
        device = 'cuda:0'
    assert resolve_device(Fake(), None).type == 'cuda'

    class CpuFake:
        device = 'cpu'
    assert resolve_device(CpuFake(), None).type == 'cuda'


def test_pipeline_refuses_to_land_on_cpu_by_accident():
    '''
    End to end: a CPU model with no device argument must still land on the GPU.

    A model sitting on the CPU does not mean "train on the CPU". The pipeline
    used to follow the model and stay on the CPU, so a script that forgot
    `model.cuda()` would silently train on the CPU for hours before anyone
    noticed. `device='auto'` now takes the GPU whenever CUDA is available, and
    only `device='cpu'` stays on the CPU.
    '''
    if not CUDA:
        return
    from motif.model.config import MotifChordConfig
    from motif.model.motif import MotifChordModel
    from motif.pipeline import PretrainConfig, PretrainPipeline

    torch.manual_seed(0)
    model = MotifChordModel(MotifChordConfig(
        vocab_size=64, model_dim=32, num_heads=4, num_kv_heads=2,
        head_k_dim=8, block=('gdn', 'chord'), block_n=1))
    pipe = PretrainPipeline(model, tokenizer=None, config=PretrainConfig(
        base_context=16, target_context=16, global_batch_tokens=16, step_mode='min',
        save_every_steps=0, probe_every_steps=0, use_progress=False, amp=False,
        eos_token_id=1))
    assert pipe.device.type == 'cuda', \
        'the pipeline must not read a CPU model as wanting CPU training'
    pipe.setup()
    assert next(pipe.model.parameters()).is_cuda, \
        'setup() must move the model onto the resolved device'


def _tiny_chord(block_n=2, dim=64, vocab_size=256):
    from motif.model.config import MotifChordConfig
    from motif.model.motif import MotifChordModel
    torch.manual_seed(0)
    return MotifChordModel(MotifChordConfig(
        vocab_size=vocab_size, model_dim=dim, num_heads=4, num_kv_heads=2,
        head_k_dim=16, block=('gdn', 'chord'), block_n=block_n))


def test_custom_ops_survive_bf16_autocast():
    '''
    The custom GDN / chord Functions must run both forward and backward under bf16 autocast.

    AMP only dispatches precision for aten operators, so inside a custom
    `torch.autograd.Function` the body mixes bf16 and fp32 and backward dies
    with `expected scalar type BFloat16 but found Float`. `autocast_safe` in
    `motif/ops/autocast.py` pins the Function body to "autocast off" and computes
    in one precision taken from the inputs.

    Also asserts that the logits come back as bf16 (so autocast really took
    effect and saved memory) while the gradients stay fp32 (standard AMP
    behaviour, the optimizer precision is unaffected).
    '''
    if not CUDA:
        return
    model = _tiny_chord().cuda()
    ids = torch.randint(0, 256, (2, 32), device='cuda')

    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        out = model(input_ids=ids)
        assert out.logits.dtype == torch.bfloat16, 'autocast did not take effect'
        loss = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, 256), ids.reshape(-1))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(g.dtype == torch.float32 for g in grads), \
        f'gradients should stay fp32, got {set(g.dtype for g in grads)}'
    assert torch.isfinite(loss)


def test_autocast_and_fp32_agree_on_scale():
    '''
    The bf16 autocast and pure fp32 losses should agree within a sane range
    (not NaN, not off by an order of magnitude).
    '''
    if not CUDA:
        return
    ids = torch.randint(0, 256, (2, 32), device='cuda')

    def run(amp):
        model = _tiny_chord().cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = model(input_ids=ids)
            return float(torch.nn.functional.cross_entropy(
                out.logits.reshape(-1, 256), ids.reshape(-1)))

    fp32_loss, amp_loss = run(False), run(True)
    assert abs(fp32_loss - amp_loss) < 0.2, \
        f'bf16 autocast and fp32 losses differ too much: {fp32_loss:.4f} vs {amp_loss:.4f}'


def test_pipeline_trains_with_amp_end_to_end():
    if not CUDA:
        return
    from motif.data import TextFileDataset
    from motif.pipeline import PretrainConfig, PretrainPipeline
    from motif.pipeline.callback import Callback
    from motif.tokenizer.tokenizer import create_tokenizer_trainer
    from motif.utils.session import TEMPLATE_PATH

    corpus = [f'the quick brown fox jumps over the lazy dog {i} 你好 世界' for i in range(64)]
    trainer = create_tokenizer_trainer(vocab_size=512)
    trainer.train_from_iterator(iter(corpus))
    tokenizer = trainer.packed_tokenizer
    tokenizer.set_chat_template(open(TEMPLATE_PATH, encoding='utf-8').read())

    class StopAfter(Callback):
        def __init__(self, n): self.n = n

        def on_step_end(self, pipeline, metrics):
            if pipeline.state.global_step >= self.n:
                self.stop_requested = True

    base = tempfile.mkdtemp(prefix='motif_amp_')
    try:
        root = os.path.join(base, 'corpus'); os.makedirs(root)
        for i in range(3):
            with open(os.path.join(root, f'{i}.txt'), 'w', encoding='utf-8') as f:
                f.write(' '.join(f'w{j}' for j in range(80)))

        model = _tiny_chord(block_n=2, dim=64, vocab_size=tokenizer.vocab_size)
        pipe = PretrainPipeline(
            model, tokenizer,
            PretrainConfig(base_context=16, target_context=32, global_batch_tokens=32,
                           step_mode='min', learning_rate=1e-3, warmup_steps=1,
                           ckpt_dir=os.path.join(base, 'ck'), save_every_steps=0,
                           probe_every_steps=0, use_progress=False, amp=True),
            callbacks=[StopAfter(3)], seed=0,
        )
        pipe.train(TextFileDataset(root, tokenizer=tokenizer, seq_len=16, drop_last=False),
                   num_epochs=1)

        assert pipe.device.type == 'cuda'
        assert pipe.state.global_step == 3
        params = list(model.parameters())
        assert all(p.dtype == torch.float32 for p in params), \
            'under AMP the parameters must stay fp32'
        assert isinstance(pipe._optimizer, torch.optim.AdamW)
        losses = pipe.metrics.get('loss/train')
        assert len(losses) == 3 and all(v == v for v in losses), f'NaN in the loss: {losses}'
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'--- {name}')
            fn()
    print('all device/amp tests passed')
