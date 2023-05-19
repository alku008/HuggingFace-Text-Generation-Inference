import os
import torch

from datetime import timedelta

class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def barrier(self, *args, **kwargs):
        return FakeBarrier()

    def size(self):
        return 1

    def rank(self):
        return 0


def initialize_torch_distributed():
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        from torch.distributed import ProcessGroupNCCL

        # Set the device id.
        assert world_size <= torch.cuda.device_count(), "Each process is one gpu"
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        backend = "nccl"
        options = ProcessGroupNCCL.Options()
        options.is_high_priority_stream = True
        options._timeout = timedelta(seconds=60)
    else:
        backend = "gloo"
        options = None

    if world_size == 1:
        return FakeGroup(), rank, world_size
    else:
        # Call the init process.
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=60),
            pg_options=options,
        )

        return torch.distributed.group.WORLD, rank, world_size
