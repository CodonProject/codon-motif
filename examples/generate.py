'''
Generation, three ways: plain text, raw tokens, and a batch with your own sampler.

    python examples/generate.py --ckpt sft.pt --prompt "从前有座山，"
    python examples/generate.py --ckpt base.pt --prompt "The capital of France is" \
        --max-new-tokens 64 --temperature 0.0
    python examples/generate.py --ckpt sft.pt --prompt "你好" --tokens-only --json

`TextGeneration` is the text-in / text-out wrapper: iterating it yields the next piece
of text, and `gen.text` holds everything produced so far. Plain `Generation` (reached
through `model.generate`) yields token tensors instead and takes a whole batch, which
is what you want when several prompts have to run together.
'''

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples._common import add_common_arguments, load_model, load_tokenizer

from motif.model.generate import TextGeneration
from motif.model.sampler import Sampler


def parse_args() -> argparse.Namespace:
    '''
    Returns:
        argparse.Namespace: The command line of this example.
    '''
    parser = argparse.ArgumentParser(description='generate text with codon-motif')
    parser.add_argument('--ckpt', required=True, help='checkpoint to generate with')
    parser.add_argument('--prompt', required=True, help='prompt text')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='0 means greedy; any value > 0 samples')
    parser.add_argument('--top-k', type=int, default=0, help='0 disables top-k')
    parser.add_argument('--top-p', type=float, default=0.0, help='0 disables nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tokens-only', action='store_true',
                        help='use plain Generation and print token ids')
    parser.add_argument('--json', action='store_true',
                        help='with --tokens-only: print JSON lines instead of a text table')
    add_common_arguments(parser)
    return parser.parse_args()


def build_sampler(args: argparse.Namespace) -> Sampler:
    '''
    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Sampler: The sampler described on the command line (greedy when the
        temperature is 0).
    '''
    return Sampler(
        temperature=args.temperature,
        top_k=args.top_k or None,
        top_p=args.top_p or None,
        repetition_penalty=args.repetition_penalty,
    )


def stream_text(model, tokenizer, args, sampler) -> None:
    '''
    Print the continuation of `--prompt` as it is produced.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        args (argparse.Namespace): Parsed arguments.
        sampler (Sampler): Sampler for this run.
    '''
    gen = TextGeneration(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        sampler=sampler,
    )
    started = time.time()
    print(f'prompt: {args.prompt!r}\nanswer: ', end='', flush=True)
    for chunk in gen:
        print(chunk, end='', flush=True)
    elapsed = max(time.time() - started, 1e-6)
    steps = gen.steps
    print(f'\n\n[{steps} tokens in {elapsed:.1f}s, {steps / elapsed:.1f} tok/s]')


def stream_tokens(model, tokenizer, args, sampler) -> None:
    '''
    Run plain `Generation` and print the token ids of every step.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        args (argparse.Namespace): Parsed arguments.
        sampler (Sampler): Sampler for this run.
    '''
    ids = tokenizer.encode(args.prompt)
    prompt = torch.tensor([list(ids)], dtype=torch.long, device=next(model.parameters()).device)
    if not args.json:
        print(f'prompt ids ({len(ids)}): {list(ids)}')

    gen = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.token_eos,
        sampler=sampler,
        tokenizer=tokenizer,
    )
    started = time.time()
    rows = []
    for step, token in enumerate(gen):
        token_id = int(token.item())
        piece = tokenizer.decode([token_id])
        rows.append((step, token_id, piece))
        if args.json:
            print(json.dumps({'step': step, 'id': token_id, 'piece': piece},
                             ensure_ascii=False))
    elapsed = max(time.time() - started, 1e-6)

    if not args.json:
        for step, token_id, piece in rows:
            print(f'{step:>4}  {token_id:>6}  {piece!r}')
    print(f'[{gen.steps} tokens in {elapsed:.1f}s, {gen.steps / elapsed:.1f} tok/s, '
          f'finished={bool(gen.finished.all())}]', file=sys.stderr)


def main() -> int:
    '''
    Returns:
        int: Process exit code.
    '''
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args)
    model, _ = load_model(args.ckpt, args.device)
    sampler = build_sampler(args)

    if args.tokens_only:
        stream_tokens(model, tokenizer, args, sampler)
    else:
        stream_text(model, tokenizer, args, sampler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
