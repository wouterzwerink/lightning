import argparse
import os

import deepspeed
import torch
import torch.nn as nn


class TheModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer = torch.nn.Linear(32, 2, bias=False)
        self.layer = nn.Conv2d(9, 9, 3)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


config = {
    "activation_checkpointing": {
        "contiguous_memory_optimization": False,
        "cpu_checkpointing": False,
        "partition_activations": False,
        "synchronize_checkpoint_boundary": False,
    },
    "aio": {
        "block_size": 1048576,
        "overlap_events": True,
        "queue_depth": 8,
        "single_submit": False,
        "thread_count": 1,
    },
    "train_micro_batch_size_per_gpu": 1,
    "zero_allow_untested_optimizer": True,
    "zero_optimization": {
        "allgather_bucket_size": 200000000,
        "allgather_partitions": True,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_bucket_size": 200000000,
        "reduce_scatter": True,
        "stage": 3,
        "sub_group_size": 1000000000000,
    },
}


def worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12234"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    deepspeed.init_distributed()

    model_parallel_context = deepspeed.zero.Init(
        remote_device="cpu", pin_memory=True, config_dict_or_path=config, dtype=torch.float32
    )

    with model_parallel_context:
        model = TheModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=rank),
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            dist_init_required=False,
            config=config,
        )


if __name__ == "__main__":
    torch.multiprocessing.spawn(worker, nprocs=2)
