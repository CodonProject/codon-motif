import json
import os
import shutil
import tempfile

import torch
import torch.nn as nn

from motif.data import MotifSFT
from motif.pipeline import Callback, SFTConfig, SFTPipeline
from motif.utils.lora import (
    LoRAConfig,
    apply_lora,
    iter_lora_modules,
    load_lora,
    lora_parameter_count,
    lora_state_dict,
    lora_summary,
    mark_only_lora_as_trainable,
    merge_lora,
    save_lora,
    unmerge_lora,
)

from support import make_model, make_tokenizer


def make_inputs(vocab_size: int, batch: int = 2, length: int = 12) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randint(1, vocab_size, (batch, length), generator=generator)


def assert_raises(exception, function, *args, **kwargs) -> str:
    '''Assert that `function` raises `exception` and return the exception message.'''
    try:
        function(*args, **kwargs)
    except exception as error:
        return str(error)
    raise AssertionError(f'{exception.__name__} was not raised')


def perturb_adapter(model: nn.Module, scale: float = 0.05) -> None:
    '''Move every adapter away from its initial value (the initial lora_b is always 0, so without perturbation no difference can be measured).'''
    with torch.no_grad():
        for index, (_, module) in enumerate(iter_lora_modules(model)):
            module.lora_b.data.normal_(0.0, scale * (1.0 + 0.1 * index))


def freeze_snapshot(model: nn.Module) -> dict:
    '''Snapshot of the non-adapter parameters, used to prove the base model was left untouched.'''
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if 'lora_' not in name
    }


def test_config_rejects_invalid_geometry():
    assert 'r must be >= 1' in assert_raises(ValueError, LoRAConfig, r=0)
    assert 'alpha must be > 0' in assert_raises(ValueError, LoRAConfig, alpha=0.0)
    assert 'dropout must be in [0, 1)' in assert_raises(ValueError, LoRAConfig, dropout=1.0)


def test_config_state_round_trip():
    config = LoRAConfig(r=4, alpha=8.0, dropout=0.1, target_modules=('q_proj',), train_bias=True)
    state = config.state()
    assert json.loads(json.dumps(state)) == state
    assert state['target_modules'] == ['q_proj']
    assert LoRAConfig.from_state(state).state() == state
    assert LoRAConfig.from_state(None).state() == LoRAConfig().state()


def test_config_matches_names():
    default = LoRAConfig()
    assert default.matches('blocks.0.features.0.attn.q_proj')
    assert not default.matches('embed')
    assert not default.matches('lm_head')
    targeted = LoRAConfig(target_modules=('q_proj',))
    assert targeted.matches('blocks.0.features.0.attn.q_proj')
    assert not targeted.matches('blocks.0.features.0.attn.o_proj')


def test_config_scaling_is_alpha_over_r():
    assert LoRAConfig(r=8, alpha=16.0).scaling == 2.0


def test_apply_lora_freezes_the_base_model():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    adapter, trainable = lora_parameter_count(model)
    assert adapter == 0
    assert trainable > 0

    config = apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    assert isinstance(config, LoRAConfig)

    layers = list(iter_lora_modules(model))
    assert layers, 'nothing was wrapped'
    assert all(name.endswith('_proj') for name, _ in layers)
    assert not any(name.startswith(('embed', 'lm_head')) for name, _ in layers)
    for name, param in model.named_parameters():
        assert param.requires_grad == ('lora_' in name), name


def test_apply_lora_only_wraps_the_targeted_modules():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    plain = sum(1 for name, module in model.named_modules()
                if isinstance(module, nn.Linear) and name.endswith('.q_proj'))
    apply_lora(model, LoRAConfig(r=4, alpha=8.0, target_modules=('q_proj',)))
    names = [name for name, _ in iter_lora_modules(model)]
    assert len(names) == plain > 0
    assert all(name.endswith('.q_proj') for name in names)


def test_apply_lora_is_identity_at_initialization():
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    before = model(ids).logits.detach().clone()

    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    after = model(ids).logits
    assert torch.equal(before, after)


def test_apply_lora_mirrors_the_host_mode():
    '''A LoRALinear created by `apply_lora` defaults to train mode; when the host is already
    in eval it must follow suit, otherwise an adapter with dropout would break the
    determinism of inference.'''
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    apply_lora(model, LoRAConfig(r=4, alpha=8.0, dropout=0.5))
    assert all(not module.training for _, module in iter_lora_modules(model))

    ids = make_inputs(tokenizer.vocab_size)
    with torch.no_grad():
        first = model(ids).logits.clone()
        second = model(ids).logits
    assert torch.equal(first, second)

    model.train()
    assert all(module.training for _, module in iter_lora_modules(model))


def test_merge_matches_the_adapter_forward_with_dropout_enabled():
    '''With dropout > 0, the eval-mode adapter forward must match the forward after merge.'''
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    ids = make_inputs(tokenizer.vocab_size)
    apply_lora(model, LoRAConfig(r=4, alpha=8.0, dropout=0.3))
    perturb_adapter(model)

    with torch.no_grad():
        adapted = model(ids).logits.clone()
    merge_lora(model)
    with torch.no_grad():
        merged = model(ids).logits
    assert torch.allclose(adapted, merged, atol=1e-4)


def test_tied_weights_survive_wrapping():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    assert model.embed.weight is model.lm_head.weight
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    assert model.embed.weight is model.lm_head.weight
    assert not any(name.startswith(('embed', 'lm_head')) for name, _ in iter_lora_modules(model))


def test_lora_factors_land_next_to_the_wrapped_layer():
    '''`apply_lora` is often called only after the model is already on the device
    (`SFTPipeline.setup` does exactly that), so the new parameters must share the device
    and dtype of the base model, otherwise the first forward pass fails with "Expected all
    tensors to be on the same device".'''
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size)
    if torch.cuda.is_available():
        model = model.to('cuda')
        ids = ids.to('cuda')

    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    for name, module in iter_lora_modules(model):
        assert module.lora_a.device == module.base.weight.device, name
        assert module.lora_b.device == module.base.weight.device, name
        assert module.lora_b.dtype == module.base.weight.dtype, name

    model.eval()
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape[:2] == ids.shape
    assert torch.isfinite(logits).all()


def test_apply_lora_without_a_match_raises():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    message = assert_raises(
        ValueError, apply_lora, model, LoRAConfig(target_modules=('no_such_proj',))
    )
    assert 'no nn.Linear matched' in message


def test_apply_lora_rejects_a_foreign_config():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    assert 'must be a LoRAConfig' in assert_raises(TypeError, apply_lora, model, {'r': 4})


def test_adapter_state_dict_holds_only_lora_keys():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    state = lora_state_dict(model)
    layers = list(iter_lora_modules(model))
    assert len(state) == 2 * len(layers)
    assert all(name.endswith(('.lora_a', '.lora_b')) for name in state)
    assert all(tensor.device.type == 'cpu' for tensor in state.values())


def test_mark_only_lora_as_trainable_refreezes_everything_else():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    for param in model.parameters():
        param.requires_grad_(True)
    mark_only_lora_as_trainable(model)
    adapter, trainable = lora_parameter_count(model)
    assert adapter > 0
    assert adapter == trainable
    assert len(lora_state_dict(model)) == 2 * len(list(iter_lora_modules(model)))


def test_summary_reports_the_geometry():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    assert lora_summary(model) == ''
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    summary = lora_summary(model)
    assert summary.startswith('LoRA r=4 alpha=8 layers=')
    adapter, trainable = lora_parameter_count(model)
    assert f'adapter_params={adapter:,}' in summary
    assert f'trainable={trainable:,}' in summary


def test_merge_matches_the_adapter_forward_path():
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    perturb_adapter(model)
    before = model(ids).logits.detach().clone()

    merge_lora(model)
    after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4)


def test_unmerge_restores_the_adapter_forward_path():
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    perturb_adapter(model)
    before = model(ids).logits.detach().clone()

    merge_lora(model)
    unmerge_lora(model)
    assert torch.allclose(before, model(ids).logits, atol=1e-4, rtol=1e-4)
    assert not any(module.merged for _, module in iter_lora_modules(model))


def test_merge_twice_is_idempotent():
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    perturb_adapter(model)
    merge_lora(model)
    once = model(ids).logits.detach().clone()
    merge_lora(model)
    assert torch.allclose(once, model(ids).logits, atol=1e-6, rtol=1e-6)


def test_merge_unload_returns_a_plain_model():
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    perturb_adapter(model)
    expected = model(ids).logits.detach().clone()

    merge_lora(model, unload=True)
    assert list(iter_lora_modules(model)) == []
    assert 'lora_a' not in model.state_dict()
    assert torch.allclose(expected, model(ids).logits, atol=1e-4, rtol=1e-4)


def _round_trip(suffix: str) -> None:
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    directory = tempfile.mkdtemp()
    try:
        model = make_model(vocab_size=tokenizer.vocab_size).eval()
        config = LoRAConfig(r=4, alpha=8.0, dropout=0.0)
        apply_lora(model, config)
        perturb_adapter(model)
        expected = model(ids).logits.detach().clone()

        path = os.path.join(directory, 'adapter' + suffix)
        saved = save_lora(model, path, config=config, metadata={'stage': 'tiny'})
        assert os.path.exists(path)
        assert set(saved) == set(lora_state_dict(model))

        fresh = make_model(vocab_size=tokenizer.vocab_size).eval()
        apply_lora(fresh, config)
        loaded = load_lora(fresh, path)
        assert loaded.state() == config.state()
        assert torch.equal(expected, fresh(ids).logits)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_safetensors_round_trip():
    _round_trip('.safetensors')


def test_torch_round_trip():
    _round_trip('.pt')


def test_load_lora_accepts_a_plain_adapter_dict():
    tokenizer = make_tokenizer()
    ids = make_inputs(tokenizer.vocab_size)
    model = make_model(vocab_size=tokenizer.vocab_size).eval()
    config = apply_lora(model, LoRAConfig(r=4, alpha=8.0))
    perturb_adapter(model)
    expected = model(ids).logits.detach().clone()

    fresh = make_model(vocab_size=tokenizer.vocab_size).eval()
    apply_lora(fresh, config)
    load_lora(fresh, lora_state_dict(model))
    assert torch.equal(expected, fresh(ids).logits)


def test_load_lora_strict_rejects_a_mismatched_configuration():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        model = make_model(vocab_size=tokenizer.vocab_size)
        apply_lora(model, LoRAConfig(r=4, alpha=8.0, target_modules=('q_proj',)))
        path = os.path.join(directory, 'narrow.safetensors')
        save_lora(model, path, config=LoRAConfig(target_modules=('q_proj',)))

        fresh = make_model(vocab_size=tokenizer.vocab_size)
        apply_lora(fresh, LoRAConfig(r=4, alpha=8.0))
        message = assert_raises(RuntimeError, load_lora, fresh, path)
        assert 'adapter tensor(s) are missing from the file' in message
        load_lora(fresh, path, strict=False)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_save_lora_without_an_adapter_raises():
    tokenizer = make_tokenizer()
    model = make_model(vocab_size=tokenizer.vocab_size)
    message = assert_raises(ValueError, save_lora, model, os.path.join(tempfile.gettempdir(), 'x.pt'))
    assert 'no LoRA adapter' in message


def test_save_lora_creates_the_parent_directory():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        model = make_model(vocab_size=tokenizer.vocab_size)
        config = apply_lora(model, LoRAConfig(r=2, alpha=4.0))
        path = os.path.join(directory, 'nested', 'deep', 'adapter.pt')
        save_lora(model, path, config=config)
        assert os.path.exists(path)
        payload = torch.load(path, weights_only=False)
        assert payload['__lora__'] is True
        assert payload['lora_spec']['r'] == 2
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def write_sft_folder(directory: str) -> str:
    '''Write a small input/cot/content jsonl (cot may be an empty string).'''
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'part.jsonl'), 'w', encoding='utf-8') as handle:
        for i in range(12):
            row = {'input': f'question {i}?', 'cot': f'compute {i}', 'content': f'answer {i}'}
            if i % 4 == 0:
                row['cot'] = ''
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return directory


def make_sft_config(folder: str, ckpt_dir: str, **overrides) -> SFTConfig:
    kwargs = dict(
        compiled=False,
        learning_rate=1e-3,
        warmup_steps=1,
        ckpt_dir=ckpt_dir,
        save_every_steps=0,
        auto_resume=False,
        use_progress=False,
        amp=False,
        stage_specs=[{'name': 'tiny', 'folder': folder, 'epochs': 1}],
        pad_length=48,
        batch_size=2,
    )
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


class StopAfter(Callback):
    '''Request a stop once n optimization steps have completed.'''

    def __init__(self, n):
        self.n = n

    def on_step_end(self, pipeline, step_metrics):
        if pipeline.state.global_step >= self.n:
            self.stop_requested = True


def make_pipeline(folder: str, ckpt_dir: str, tokenizer, callbacks=None, **overrides) -> SFTPipeline:
    config = make_sft_config(folder, ckpt_dir, **overrides)
    if callbacks is None:
        callbacks = [StopAfter(2)]
    return SFTPipeline(
        make_model(vocab_size=tokenizer.vocab_size), tokenizer, config,
        callbacks=callbacks, seed=0, device='cpu',
    )


def test_sft_with_lora_trains_only_the_adapter():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        model = make_model(vocab_size=tokenizer.vocab_size)
        before = freeze_snapshot(model)
        assert lora_state_dict(model) == {}

        lora = LoRAConfig(r=4, alpha=8.0)
        matched = {name for name, module in model.named_modules()
                   if isinstance(module, nn.Linear) and name and lora.matches(name)}
        assert matched and not any(name.startswith(('embed', 'lm_head')) for name in matched)

        config = make_sft_config(
            folder, os.path.join(directory, 'ck'), lora=lora
        )
        pipe = SFTPipeline(
            model, tokenizer, config, callbacks=[StopAfter(2)], seed=0, device='cpu'
        )
        metrics = pipe.train()
        assert pipe.lora_config is not None
        assert pipe.state.global_step == 2
        assert 'loss/train' in metrics

        after = dict(pipe.model.named_parameters())
        for name, tensor in before.items():
            head, _, tail = name.rpartition('.')
            wrapped = f'{head}.base.{tail}' if head in matched else name
            assert wrapped in after, wrapped
            gap = (tensor - after[wrapped].detach()).abs().max().item()
            assert torch.equal(tensor, after[wrapped].detach()), f'{name} max_diff={gap}'
        assert all(param.requires_grad for name, param in after.items() if 'lora_' in name)
        assert not any(param.requires_grad for name, param in after.items() if 'lora_' not in name)

        adapters = lora_state_dict(pipe.model)
        assert adapters
        assert any(tensor.abs().sum() > 0 for name, tensor in adapters.items()
                   if name.endswith('.lora_b'))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_payload_carries_the_adapter_not_the_model():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        pipe = make_pipeline(
            folder, os.path.join(directory, 'ck'), tokenizer,
            lora=LoRAConfig(r=4, alpha=8.0),
        )
        pipe.setup()
        payload = pipe.state_payload()
        assert 'adapter' in payload and 'model' not in payload
        assert payload['lora_spec']['r'] == 4
        assert set(payload['adapter']) == set(lora_state_dict(pipe.model))
        assert payload['optimizer'] is not None and payload['scheduler'] is not None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_without_lora_still_stores_the_whole_model():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        pipe = make_pipeline(folder, os.path.join(directory, 'ck'), tokenizer)
        pipe.setup()
        payload = pipe.state_payload()
        assert 'model' in payload and 'adapter' not in payload
        assert pipe.lora_config is None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_stage_export_writes_a_loadable_adapter():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        ckpt = os.path.join(directory, 'export', 'tiny_adapter.safetensors')
        pipe = make_pipeline(
            folder, os.path.join(directory, 'ck'), tokenizer,
            callbacks=[],
            lora=LoRAConfig(r=4, alpha=8.0),
            stage_specs=[{'name': 'tiny', 'folder': folder, 'epochs': 1, 'ckpt': ckpt}],
        )
        pipe.train()
        assert os.path.exists(ckpt)

        fresh = make_model(vocab_size=tokenizer.vocab_size).eval()
        apply_lora(fresh, LoRAConfig(r=4, alpha=8.0))
        spec = load_lora(fresh, ckpt)
        assert spec.r == 4 and spec.alpha == 8.0
        assert set(lora_state_dict(fresh)) == set(lora_state_dict(pipe.model))
        for name, tensor in lora_state_dict(fresh).items():
            assert torch.equal(tensor, lora_state_dict(pipe.model)[name])
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_resumes_from_an_adapter_checkpoint():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        ckpt = os.path.join(directory, 'resume.pt')
        first = make_pipeline(
            folder, os.path.join(directory, 'ck1'), tokenizer,
            lora=LoRAConfig(r=4, alpha=8.0),
        )
        first.setup()
        perturb_adapter(first.model)
        expected = lora_state_dict(first.model)
        first.save_checkpoint(ckpt)
        assert os.path.exists(ckpt)

        second = make_pipeline(
            folder, os.path.join(directory, 'ck2'), tokenizer,
            lora=LoRAConfig(r=4, alpha=8.0),
        )
        second.load_checkpoint(ckpt)
        second.setup()
        restored = lora_state_dict(second.model)
        assert set(restored) == set(expected)
        for name, tensor in expected.items():
            assert torch.equal(tensor, restored[name]), name
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_refuses_an_adapter_checkpoint_without_the_lora_config():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        ckpt = os.path.join(directory, 'adapter_only.pt')
        first = make_pipeline(
            folder, os.path.join(directory, 'ck1'), tokenizer,
            lora=LoRAConfig(r=4, alpha=8.0),
        )
        first.setup()
        first.save_checkpoint(ckpt)

        plain = make_pipeline(folder, os.path.join(directory, 'ck2'), tokenizer)
        plain.load_checkpoint(ckpt)
        message = assert_raises(RuntimeError, plain.setup)
        assert 'holds a LoRA adapter but config.lora is off' in message
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_rejects_an_invalid_lora_spec():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        pipe = make_pipeline(folder, os.path.join(directory, 'ck'), tokenizer, lora='yes')
        message = assert_raises(ValueError, pipe.setup)
        assert 'lora must be None, True or a LoRAConfig' in message
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_lora_true_uses_the_default_configuration():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        pipe = make_pipeline(folder, os.path.join(directory, 'ck'), tokenizer, lora=True)
        pipe.setup()
        assert pipe.lora_config == LoRAConfig()
        assert 'LoRA r=8 alpha=16' in '\n'.join(pipe.describe())
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_sft_describe_mentions_the_adapter():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        pipe = make_pipeline(
            folder, os.path.join(directory, 'ck'), tokenizer,
            lora=LoRAConfig(r=4, alpha=8.0),
        )
        pipe.setup()
        text = '\n'.join(pipe.describe())
        assert 'lora' in text and 'LoRA r=4 alpha=8' in text
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_motif_sft_limit_caps_every_file():
    tokenizer = make_tokenizer()
    directory = tempfile.mkdtemp()
    try:
        folder = write_sft_folder(os.path.join(directory, 'data'))
        rows = MotifSFT._load_jsonl(folder, '*.jsonl', True, limit=5)
        assert len(rows) == 5
        assert rows[0]['input'] == 'question 0?' and rows[-1]['input'] == 'question 4?'
        assert len(MotifSFT._load_jsonl(folder, '*.jsonl', True)) == 12

        full = MotifSFT(folder, tokenizer, pad_length=48, batch_size=2)
        capped = MotifSFT(folder, tokenizer, pad_length=48, batch_size=2, limit=5)
        assert len(full.samples) == 12
        assert len(capped.samples) == 5
        seen = {json.dumps(row, sort_keys=True) for row in full.samples}
        assert all(json.dumps(row, sort_keys=True) in seen for row in capped.samples)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'[OK] {name}')
    print('\nall LoRA tests passed')
