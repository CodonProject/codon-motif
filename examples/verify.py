'''
Check that this install and a checkpoint agree: environment, config, weights,
forward, generation and the Triton op paths.

    python examples/verify.py --ckpt D:/MotifChord/sft_test.pt
    python examples/verify.py --ckpt ckpt/last.pt --prompt "你好" --tokens 32

Everything this script prints is measured on the checkpoint it was given; a failure
raises instead of printing a warning, so the exit code is the answer. Run it after
`pip install`, after copying weights between machines, or when a generation looks
wrong and you want to know whether the model or the plumbing is to blame.
'''

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples._common import add_common_arguments, load_model, load_tokenizer

from motif.model.checkpoint import load_checkpoint
from motif.model.config import MotifChordConfig
from motif.model.generate import TextGeneration
from motif.model.motif import MotifChordModel
from motif.model.sampler import Sampler
from motif.ops import HAS_TRITON, can_use_triton_attn_res
from motif.utils.units import format_count


def parse_args() -> argparse.Namespace:
    '''
    Returns:
        argparse.Namespace: The command line of this example.
    '''
    parser = argparse.ArgumentParser(description='verify a checkpoint end to end')
    parser.add_argument('--ckpt', required=True, help='checkpoint to verify')
    parser.add_argument('--prompt', default='从前有座山，', help='generation probe')
    parser.add_argument('--tokens', type=int, default=16, help='tokens of that probe')
    parser.add_argument('--warmup', type=int, default=1, help='untimed decode steps')
    parser.add_argument('--bench-steps', type=int, default=0,
                        help='extra timed decode steps (0 = only the probe)')
    add_common_arguments(parser)
    return parser.parse_args()


def report_environment() -> None:
    '''Print the versions that decide which op path a run takes.'''
    print('== environment ==')
    print(f'torch        {torch.__version__}')
    print(f'cuda         {torch.version.cuda}, available={torch.cuda.is_available()}, '
          f'devices={torch.cuda.device_count()}')
    if torch.cuda.is_available():
        print(f'device       {torch.cuda.get_device_name(0)}')
    print(f'triton       {"yes" if HAS_TRITON else "no"}')
    try:
        import triton
        print(f'triton ver   {triton.__version__}')
    except ImportError:
        pass


def report_config(model: MotifChordModel) -> None:
    '''
    Print the resolved architecture.

    Args:
        model (MotifChordModel): The loaded model.
    '''
    config = model.config
    print('== config ==')
    for field in ('vocab_size', 'model_dim', 'num_heads', 'num_kv_heads', 'head_k_dim',
                  'block', 'block_n', 'tied_weight'):
        print(f'{field:<13}{getattr(config, field)}')
    print(f'{"vision":<13}{getattr(config, "vision", False)}')


def report_weights(args: argparse.Namespace, model: MotifChordModel) -> None:
    '''
    Re-load the file under the resolved config and report every mismatch.

    Args:
        args (argparse.Namespace): Parsed arguments.
        model (MotifChordModel): The loaded model.
    '''
    print('== weights ==')
    loaded = load_checkpoint(args.ckpt, map_location='cpu')
    print(f'file         {loaded.path}')
    print(f'metadata     {loaded.metadata}')
    print(f'tensors      {len(loaded.state_dict)}')
    stored = sum(t.numel() for t in loaded.state_dict.values())
    params = sum(p.numel() for p in model.parameters())
    print(f'stored       {format_count(stored)} parameters')
    print(f'live         {format_count(params)} parameters')
    print(f'embedded     {model.embed.weight.shape}, '
          f'tied={model.embed.weight.data_ptr() == model.lm_head.weight.data_ptr()}')

    result = model.load_state_dict(loaded.state_dict, strict=False)
    print(f'missing      {len(result.missing_keys)} {result.missing_keys[:4]}')
    print(f'unexpected   {len(result.unexpected_keys)} {result.unexpected_keys[:4]}')
    vision = [k for k in loaded.state_dict if k.startswith('vision.')]
    print(f'vision keys  {len(vision)}')


def report_forward(model: MotifChordModel, tokenizer, device: torch.device) -> None:
    '''
    Run one forward pass and check the shapes.

    Args:
        model (MotifChordModel): The loaded model.
        tokenizer: The loaded tokenizer.
        device (torch.device): Device of the model.
    '''
    print('== forward ==')
    ids = tokenizer.encode('hello world')
    batch = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(batch)
    print(f'input        {tuple(batch.shape)}')
    print(f'logits       {tuple(out.logits.shape)}')
    print(f'hidden       {tuple(out.last_hidden_state.shape)}')
    print(f'finite       {bool(torch.isfinite(out.last_logits).all())}')
    print(f'cache types  {[[t.__name__ for t in row] for row in model.cache_types()]}')


def report_generate(model: MotifChordModel, tokenizer, args: argparse.Namespace) -> None:
    '''
    Generate `--tokens` tokens, then measure the decode speed if asked.

    Args:
        model (MotifChordModel): The loaded model.
        tokenizer: The loaded tokenizer.
        args (argparse.Namespace): Parsed arguments.
    '''
    print('== generate ==')
    greedy = Sampler(temperature=0.0, repetition_penalty=1.0)
    gen = TextGeneration(model, tokenizer, args.prompt, max_new_tokens=args.tokens,
                         sampler=greedy)
    started = time.time()
    for _ in gen:
        pass
    elapsed = max(time.time() - started, 1e-6)
    print(f'prompt       {args.prompt!r}')
    print(f'answer       {gen.text.strip()!r}')
    print(f'speed        {gen.steps} tokens in {elapsed:.2f}s '
          f'({gen.steps / elapsed:.1f} tok/s, greedy)')

    if args.bench_steps:
        budget = args.warmup + args.bench_steps
        run = TextGeneration(model, tokenizer, args.prompt, max_new_tokens=budget,
                             sampler=greedy)
        times = []
        last = time.time()
        for step, _ in enumerate(run):
            now = time.time()
            if step >= args.warmup:
                times.append(now - last)
            last = now
        if times:
            speed = 1.0 / (sum(times) / len(times))
            print(f'steady       {speed:.1f} tok/s (mean of {len(times)} steps, '
                  f'{args.warmup} warmup)')


def report_ops(device: torch.device) -> None:
    '''
    Print whether the Triton attention-res kernel can run on this box.

    Args:
        device (torch.device): Device the model runs on.
    '''
    print('== ops ==')
    print(f'chord triton available      {HAS_TRITON}')
    if not HAS_TRITON or device.type != 'cuda':
        print('(the eager fallbacks are in use on this device)')
        return

    from motif.ops.attn_res import attn_res_block
    from motif.ops.rms_norm import RMSNorm

    dim, batch, seq = 64, 2, 4
    sources = [torch.randn(batch, seq, dim, device=device) for _ in range(2)]
    weight = torch.ones(dim, device=device)
    query = torch.randn(dim, device=device)
    norm = RMSNorm(dim).to(device)
    print(f'attn-res fused usable       {can_use_triton_attn_res(sources, weight, query)}')
    out = attn_res_block(sources, weight, query, eps=1e-5)
    print(f'attn-res out                {tuple(out.shape)}, '
          f'finite={bool(torch.isfinite(out).all())}')

    from motif.ops import apply_chord_attention
    heads, kv_heads, head_dim, q_len, kv_len = 2, 1, 32, 4, 4
    q = torch.randn(batch, heads, q_len, head_dim, device=device)
    k = torch.randn(batch, kv_heads, kv_len, head_dim, device=device)
    v = torch.randn(batch, kv_heads, kv_len, head_dim, device=device)
    out = apply_chord_attention(
        q, k, v,
        torch.ones(heads, device=device), torch.zeros(heads, device=device),
        torch.zeros(heads, device=device), torch.ones(heads, device=device),
        torch.zeros(batch, q_len, device=device),
        q_start=0, softmax_scale=head_dim ** -0.5, eps=1e-5,
    )
    print(f'chord kernel out            {tuple(out.shape)}, '
          f'finite={bool(torch.isfinite(out).all())}')


def main() -> int:
    '''
    Returns:
        int: Process exit code.
    '''
    args = parse_args()
    report_environment()

    tokenizer = load_tokenizer(args)
    model, device = load_model(args.ckpt, args.device)
    report_config(model)
    report_weights(args, model)
    report_forward(model, tokenizer, device)
    report_generate(model, tokenizer, args)
    report_ops(device)

    print('\n[*] all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
