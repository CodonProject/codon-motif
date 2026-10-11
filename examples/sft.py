'''
Supervised fine-tuning on a folder of chat jsonl, optionally with LoRA adapters.

    python examples/sft.py --data data/general --ckpt sft_ckpt
    python examples/sft.py --data data/role --ckpt sft_lora --base ../MotifChord/sft_test.pt \
        --lora --lora-r 16 --epochs 1 --batch-size 4 --pad-length 1024
    python examples/sft.py --data data/general --data data/role --name general --name role \
        --epochs 2 --epochs 1 --ckpt sft_ckpt --tensorboard runs/sft

`--data` is a folder of `.jsonl` files in any of the schemas `MotifSFT` accepts
(`conversations`, `messages`, `session`, plain instruction/response pairs). Pass it
more than once - with a matching `--name` and `--epochs` - to chain stages, which is
how a short "general" pass is normally followed by a longer "style" pass.

LoRA freezes the base model and trains only low-rank adapters; the checkpoint then
holds the adapter plus its spec instead of a full copy, and `motif.utils.lora.merge_lora`
folds it back into the base weights for inference.
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples._common import add_common_arguments, load_tokenizer

from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.pipeline import SFTConfig, SFTPipeline, build_sft_stages
from motif.pipeline.distributed import launch
from motif.utils.lora import LoRAConfig


def parse_args() -> argparse.Namespace:
    '''
    Returns:
        argparse.Namespace: The command line of this example.
    '''
    parser = argparse.ArgumentParser(description='supervised fine-tuning')
    parser.add_argument('--data', action='append', required=True,
                        help='folder of chat jsonl; repeat per stage')
    parser.add_argument('--name', action='append', default=None,
                        help='stage name; repeat to match --data')
    parser.add_argument('--epochs', action='append', type=int, default=None,
                        help='epochs of one stage; repeat to match --data')
    parser.add_argument('--stage-ckpt', action='append', default=None,
                        help='export the weights here when a stage finishes')
    parser.add_argument('--ckpt', default='sft_ckpt', help='checkpoint directory')
    parser.add_argument('--base', default=None,
                        help='base checkpoint the fine-tune starts from (fresh model when unset)')
    parser.add_argument('--model-config', default=None,
                        help='JSON file with MotifChordConfig overrides for a fresh model')
    parser.add_argument('--pad-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--lora', action='store_true', help='train LoRA adapters only')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.0)
    parser.add_argument('--lora-targets', default=None,
                        help='comma separated leaf module names, e.g. q_proj,k_proj,v_proj,o_proj')
    parser.add_argument('--sanity-samples', type=int, default=0,
                        help='score (and answer) N training conversations every probe')
    parser.add_argument('--log-file', default=None)
    parser.add_argument('--tensorboard', default=None)
    parser.add_argument('--single-process', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    add_common_arguments(parser)
    return parser.parse_args()


def stage_specs(args: argparse.Namespace) -> list:
    '''
    Pair every `--data` with its name, epoch count and export path.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        list: One dict per stage, ready for `build_sft_stages`.
    '''
    names = args.name or []
    epochs = args.epochs or []
    exports = args.stage_ckpt or []
    specs = []
    for index, folder in enumerate(args.data):
        name = names[index] if index < len(names) else os.path.basename(os.path.normpath(folder))
        specs.append({
            'name': name or f'stage{index}',
            'folder': folder,
            'epochs': epochs[index] if index < len(epochs) else 1,
            'ckpt': exports[index] if index < len(exports) else None,
        })
    return specs


def build_model(args: argparse.Namespace):
    '''
    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        MotifChordModel: The base checkpoint, or a fresh model when `--base` is unset.
    '''
    if args.base:
        print(f'[*] fine-tuning from {args.base}')
        return MotifChordModel.from_pretrained(args.base, map_location='cpu')

    import json
    from pathlib import Path

    overrides = json.loads(Path(args.model_config).read_text(encoding='utf-8')) \
        if args.model_config else {}
    model = MotifChordModel(MotifChordConfig(**overrides))
    model.count_param(print=True)
    return model


def build_lora(args: argparse.Namespace):
    '''
    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        Optional[LoRAConfig]: The adapter configuration, or None for a full fine-tune.
    '''
    if not args.lora:
        return None
    targets = tuple(t.strip() for t in args.lora_targets.split(',')) if args.lora_targets else None
    return LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, target_modules=targets)


def build_config(args: argparse.Namespace, stages) -> SFTConfig:
    '''
    Args:
        args (argparse.Namespace): Parsed arguments.
        stages (list): Stages that were just built.

    Returns:
        SFTConfig: The fine-tuning configuration.
    '''
    return SFTConfig(
        lora=build_lora(args),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        pad_length=args.pad_length,
        batch_size=args.batch_size,
        ckpt_dir=args.ckpt,
        log_file=args.log_file,
        tensorboard=args.tensorboard,
        probe_prompts=['user: hello\nassistant:'] if args.sanity_samples else [],
        probe_every_steps=200 if args.sanity_samples else 0,
        sanity_samples=args.sanity_samples,
        distributed={'auto_spawn': not args.single_process,
                     'follow_env': not args.single_process},
    )


def run(args: argparse.Namespace) -> int:
    '''
    Build the stages and fine-tune.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        int: Process exit code.
    '''
    tokenizer = load_tokenizer(args)
    model = build_model(args)
    specs = stage_specs(args)
    stages = build_sft_stages(
        specs,
        tokenizer=tokenizer,
        pad_length=args.pad_length,
        batch_size=args.batch_size,
    )
    pipeline = SFTPipeline(
        model,
        tokenizer,
        build_config(args, stages),
        stages=stages,
        seed=args.seed,
    )

    for line in pipeline.describe():
        print(line)
    for stage in stages:
        print(f'[*] stage {stage.name!r}: {len(stage.dataset.groups)} conversations, '
              f'{stage.epochs} epoch(s)')
    metrics = pipeline.train()
    print(f'done: {metrics}')
    return 0


def main() -> int:
    '''
    Returns:
        int: Process exit code.
    '''
    args = parse_args()
    if args.single_process:
        return run(args)
    return launch(run, args=(args,))


if __name__ == '__main__':
    raise SystemExit(main())
