
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from motif.data.base import MotifDataset
from motif.model.config import MotifChordConfig
from motif.model.motif import MotifChordModel
from motif.pipeline import (
    Callback,
    DistributedConfig,
    PretrainConfig,
    PretrainPipeline,
)

#: Vocabulary of the toy model; the corpus is token ids, so no tokenizer is needed.
VOCAB_SIZE = 64
#: Document boundary / end-of-text ids used by the packing stream.
BOUNDARY_TOKEN_ID = 3
EOS_TOKEN_ID = 2
#: Global tokens per optimiser step.
#:
#: It is deliberately *not* a multiple of the rank count: on the 16-token stage it is
#: 2 rows, which 2 ranks split 1/1, and on the 8-token stage it is 3 rows, which 2 ranks
#: split 2/1. The uneven split is the interesting case - it is what the token-weighted
#: loss has to compensate for.
GLOBAL_BATCH_TOKENS = 34
BASE_CONTEXT = 8
TARGET_CONTEXT = 16


class ListDataset(MotifDataset):
    '''Map-style corpus of packed token id records (no tokenizer involved).'''

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class StopAfter(Callback):
    '''Ask the pipeline to wind down once the global step counter reaches `n`.'''

    def __init__(self, n):
        self.n = n

    def on_step_end(self, pipeline, step_metrics):
        if pipeline.state.global_step >= self.n:
            self.stop_requested = True


class RecordingPretrain(PretrainPipeline):
    '''
    Keeps the exact batch every step consumed and the gradient of every step.

    The gradients are the sharpest test of data parallelism: they are taken at the
    weights the rank started the step from, so two runs can only differ there by the
    order of the floating-point additions - not by the training dynamics that amplify a
    difference from step to step (AdamW's `eps` makes a near-zero gradient component
    jump when it changes sign, so weights drift far more than gradients do).
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []
        self.grads = []

    def setup(self):
        super().setup()
        optimizer = self._optimizer
        original_step = optimizer.step

        def capturing_step(*args, **kwargs):
            self.grads.append({
                name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                for name, parameter in self._model.named_parameters()
            })
            return original_step(*args, **kwargs)

        optimizer.step = capturing_step

    def compute_loss(self, batch):
        stage, inputs, labels = batch
        self.seen.append((stage.name, inputs.clone(), labels.clone()))
        return super().compute_loss(batch)


def build_records(count=64, seed=0):
    '''
    A deterministic corpus of packed token ids.

    Args:
        count (int): Number of records.
        seed (int): Seed of the generator, so every rank sees the same corpus.

    Returns:
        list: One list of token ids per record.
    '''
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(4, VOCAB_SIZE, (8 + index % 5,), generator=generator).tolist()
        for index in range(count)
    ]


def build_model(seed=0):
    '''
    A tiny deterministic model, identical on every rank.

    Args:
        seed (int): Seed of the initial weights.

    Returns:
        MotifChordModel: The model.
    '''
    torch.manual_seed(seed)
    return MotifChordModel(MotifChordConfig(
        vocab_size=VOCAB_SIZE,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        head_k_dim=8,
        block=('gdn', 'chord'),
        block_n=2,
    ))


def main() -> int:
    '''
    Runs one rank of the comparison run.

    Returns:
        int: Process exit code.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True,
                        help='directory receiving rank_<rank>.pt of every rank')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pipeline = RecordingPretrain(
        build_model(), None,
        PretrainConfig(
            base_context=BASE_CONTEXT,
            target_context=TARGET_CONTEXT,
            global_batch_tokens=GLOBAL_BATCH_TOKENS,
            step_mode='min',
            learning_rate=1e-2,
            warmup_steps=1,
            ckpt_dir=args.ckpt,
            save_every_steps=0,
            probe_every_steps=0,
            use_progress=False,
            amp=False,
            eos_token_id=EOS_TOKEN_ID,
            boundary_token_id=BOUNDARY_TOKEN_ID,
            distributed=DistributedConfig(backend='gloo'),
        ),
        device='cpu',
        callbacks=[StopAfter(args.steps)],
        seed=args.seed,
    )
    pipeline.train(ListDataset(build_records()), num_epochs=1)

    payload = {
        'rank': pipeline.rank,
        'local_rank': pipeline.local_rank,
        'world_size': pipeline.world_size,
        'distributed': pipeline.distributed,
        'global_step': pipeline.state.global_step,
        'steps_in_plan': pipeline.total_steps,
        'stage_batches': [stage.batch_size for stage in pipeline._plan.stages],
        'losses': pipeline.metrics.get('loss/train'),
        'grad_norms': pipeline.metrics.get('grad_norm'),
        'tokens': pipeline.metrics.get('tokens'),
        'shard': list(pipeline.shard_config()),
        'batches': [(stage, inputs, labels) for stage, inputs, labels in pipeline.seen],
        'grads': pipeline.grads,
        'weights': {key: value.detach().cpu() for key, value in pipeline._model.state_dict().items()},
    }
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, f'rank_{pipeline.rank}.pt')
    torch.save(payload, target)
    print(f'rank {pipeline.rank}/{pipeline.world_size}: '
          f'{pipeline.state.global_step} steps -> {target}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
