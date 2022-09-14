import torch
import torch.multiprocessing as mp
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_13
from lightning_lite.utilities.device_parser import num_cuda_devices, is_cuda_available


def worker(rank):
    print("successfully forked", rank)
    torch.cuda.set_device(rank)


def run():
    print("torch version", torch.__version__)
    print("greater than 1.13?", _TORCH_GREATER_EQUAL_1_13)

    # old function
    torch.cuda.device_count()
    # torch.cuda.is_available()

    # new function
    # print("num_cuda_devices:", num_cuda_devices())
    # print("available:", is_cuda_available())

    mp.start_processes(worker, nprocs=2, start_method="fork")


if __name__ == "__main__":
    run()
