
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from motif.pipeline import DistributedConfig, launch, setup_distributed


def build(rank, world_size, local_rank):
    '''
    One worker: join the group, prove it works, then report the rank layout.

    Args:
        rank (int): Rank handed out by `launch`.
        world_size (int): Number of workers.
        local_rank (int): Device index of this worker.
    '''
    context = setup_distributed(DistributedConfig(backend='gloo'))
    assert context.active, 'the worker was not registered as distributed'
    assert context.rank == rank, f'context rank {context.rank} != {rank}'
    assert context.world_size == world_size, 'world size mismatch'
    assert context.is_main == (rank == 0)

    total = torch.tensor([float(rank)])
    context.reduce_sum_(total)

    out_dir = os.environ['DIST_LAUNCH_OUT']
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'rank_{rank}.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'context_rank': context.rank,
            'context_world': context.world_size,
            'sum': float(total),
        }, handle)
    context.barrier()


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    launch(build, DistributedConfig(backend='gloo', num_processes=workers))
