'''
Pre-train (or continue pre-training) on a folder of text / parquet.

    python examples/pretrain.py --data corpus --ckpt ckpt/run1
    python examples/pretrain.py --data corpus --ckpt ckpt/run1 --total-steps 2000 \
        --global-batch-tokens 8192 --log-file ckpt/run1/progress.log
    python examples/pretrain.py --data shards --ckpt ckpt/run1 \
        --init-from ../MotifChord/pretrained_base.safetensors

`--data` is a directory of `.txt` files (scanned recursively with `--recursive`) or of
parquet shards. `PretrainPipeline` drills a context-length curriculum from
`--base-context` to `--target-context`; `--total-steps` bounds the run, otherwise the
recommended curriculum decides.

Re-running the same command resumes: `train()` finds `<ckpt>/last.pt` and restores the
weights, the optimizer, the learning-rate schedule and the data cursor, so no token is
trained on twice. `--init-from` is the other direction: it takes the weights only and
starts the optimizer, schedule and cursor from scratch - give it a fresh `--ckpt`.
'''

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples._common import add_common_arguments, load_tokenizer

from motif.data import TextFileDataset
from motif.pipeline import PretrainConfig, PretrainPipeline
from motif.pipeline.distributed import launch


def parse_args() -> argparse.Namespace:
    '''
    Returns:
        argparse.Namespace: The command line of this example.
    '''
    parser = argparse.ArgumentParser(description='pre-train a codon-motif model')
    parser.add_argument('--data', required=True,
                        help='folder of .txt files, or of parquet shards')
    parser.add_argument('--ckpt', default='ckpt/pretrain', help='checkpoint directory')
    parser.add_argument('--init-from', default=None,
                        help='checkpoint whose weights warm-start this run (new --ckpt!)')
    parser.add_argument('--model-config', default=None,
                        help='JSON file with MotifChordConfig overrides for a fresh model')
    parser.add_argument('--target-context', type=int, default=4096)
    parser.add_argument('--base-context', type=int, default=512)
    parser.add_argument('--global-batch-tokens', type=int, default=16384,
                        help='tokens of one optimiser step over the whole world')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--total-steps', type=int, default=None)
    parser.add_argument('--total-tokens', type=int, default=None)
    parser.add_argument('--eval-every-epochs', type=int, default=None)
    parser.add_argument('--recursive', action='store_true',
                        help='scan --data recursively (sharded corpora)')
    parser.add_argument('--parquet', action='store_true',
                        help='--data holds parquet files instead of .txt')
    parser.add_argument('--log-file', default=None, help='run log (also enables the sanity log)')
    parser.add_argument('--tensorboard', default=None, help='TensorBoard output directory')
    parser.add_argument('--snapshot-percent', type=float, default=None,
                        help='every this percent of the run, save raw weights')
    parser.add_argument('--no-probe', action='store_true',
                        help='do not generate probe samples during training')
    parser.add_argument('--single-process', action='store_true',
                        help='stay on one process even when several GPUs are visible')
    parser.add_argument('--seed', type=int, default=1234)
    add_common_arguments(parser)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    '''
    Build the model: a fresh one, or the weights of `--init-from`.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        MotifChordModel: The model to train.
    '''
    from motif.model.config import MotifChordConfig
    from motif.model.motif import MotifChordModel

    if args.init_from:
        model = MotifChordModel.from_pretrained(args.init_from, map_location='cpu')
        print(f'[*] warm start from {args.init_from}')
        return model

    overrides = json.loads(Path(args.model_config).read_text(encoding='utf-8')) \
        if args.model_config else {}
    config = MotifChordConfig(**overrides)
    model = MotifChordModel(config)
    model.count_param(print=True)
    return model


def build_dataset(args: argparse.Namespace, tokenizer) -> TextFileDataset:
    '''
    Args:
        args (argparse.Namespace): Parsed arguments.
        tokenizer: Loaded tokenizer.

    Returns:
        TextFileDataset: The corpus, cut into `--target-context` token segments.
    '''
    return TextFileDataset(
        args.data,
        recursive=args.recursive,
        tokenizer=None if args.parquet else tokenizer,
        seq_len=None if args.parquet else args.target_context,
        drop_last=False,
        encoding=None if args.parquet else 'utf-8',
    )


def build_config(args: argparse.Namespace) -> PretrainConfig:
    '''
    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        PretrainConfig: The pre-training configuration.
    '''
    return PretrainConfig(
        base_context=args.base_context,
        target_context=args.target_context,
        global_batch_tokens=args.global_batch_tokens,
        total_steps=args.total_steps,
        total_tokens=args.total_tokens,
        learning_rate=args.learning_rate,
        init_from=args.init_from,
        ckpt_dir=args.ckpt,
        log_file=args.log_file,
        tensorboard=args.tensorboard,
        snapshot_every_percent=args.snapshot_percent,
        probe_every_steps=0 if args.no_probe else 2000,
        distributed={'auto_spawn': not args.single_process,
                     'follow_env': not args.single_process},
    )


def run(args: argparse.Namespace) -> int:
    '''
    Build everything and train.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Process exit code.
    '''
    tokenizer = load_tokenizer(args)
    model = build_model(args)
    dataset = build_dataset(args, tokenizer)
    pipeline = PretrainPipeline(model, tokenizer, build_config(args), seed=args.seed)

    for line in pipeline.describe():
        print(line)
    metrics = pipeline.train(
        dataset,
        num_epochs=1,
        eval_every_epochs=args.eval_every_epochs,
    )
    print(f'done: {metrics}')
    return 0


def main() -> int:
    '''
    Returns:
        int: Process exit code.
    '''
    args = parse_args()
    print(f'[*] corpus: {os.path.abspath(args.data)}, checkpoints: {os.path.abspath(args.ckpt)}')
    if args.single_process:
        return run(args)
    return launch(run, args=(args,))


if __name__ == '__main__':
    raise SystemExit(main())
