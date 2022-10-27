import contextlib
import datetime
import inspect
import os
from functools import partial
from unittest import mock

import pytest
import torch
from tests_lite.helpers.runif import RunIf

from lightning_lite.accelerators import CPUAccelerator, CUDAAccelerator
from lightning_lite.plugins.collectives import torch_collective, TorchCollective
from lightning_lite.plugins.environments import LightningEnvironment
from lightning_lite.strategies.ddp_spawn import DDPSpawnStrategy
from lightning_lite.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_11, _TORCH_GREATER_EQUAL_1_13

if TorchCollective.is_available():
    from torch.distributed import ReduceOp
else:
    ReduceOp = mock.Mock()

skip_distributed_unavailable = pytest.mark.skipif(
    not TorchCollective.is_available(), reason="torch.distributed unavailable"
)

PASSED_TENSOR = mock.Mock()
PASSED_OBJECT = mock.Mock()


@contextlib.contextmanager
def check_destroy_group(autosetup_strategy):
    with mock.patch(
        "lightning_lite.plugins.collectives.torch_collective.TorchCollective.new_group",
        wraps=TorchCollective.new_group,
    ) as mock_new, mock.patch(
        "lightning_lite.plugins.collectives.torch_collective.TorchCollective.destroy_group",
        wraps=TorchCollective.destroy_group,
    ) as mock_destroy:
        yield

    # if autosetup_strategy, setup_environment calls `init_process_group` which is then not managed by collective
    extra = 0 if autosetup_strategy else 1
    assert (
        mock_new.call_count + extra == mock_destroy.call_count
    ), f"new_group={mock_new.call_count}, destroy_group={mock_destroy.call_count}"
    if TorchCollective.is_available():
        assert not torch.distributed.distributed_c10d._pg_map
        assert not TorchCollective.is_initialized()


@pytest.mark.parametrize(
    ["fn_name", "kwargs", "return_key"],
    [
        ("send", {"tensor": PASSED_TENSOR, "dst": 0, "tag": 0}, None),
        ("recv", {"tensor": PASSED_TENSOR, "src": 0, "tag": 0}, "tensor"),
        ("broadcast", {"tensor": PASSED_TENSOR, "src": 0}, "tensor"),
        ("all_reduce", {"tensor": PASSED_TENSOR, "op": ReduceOp.SUM}, "tensor"),
        ("reduce", {"tensor": PASSED_TENSOR, "dst": 0, "op": ReduceOp.SUM}, "tensor"),
        ("all_gather", {"tensor_list": [PASSED_TENSOR], "tensor": PASSED_TENSOR}, "tensor_list"),
        ("gather", {"tensor": PASSED_TENSOR, "gather_list": [PASSED_TENSOR], "dst": 0}, "gather_list"),
        ("scatter", {"tensor": PASSED_TENSOR, "scatter_list": [PASSED_TENSOR], "src": 0}, "tensor"),
        ("reduce_scatter", {"output": PASSED_TENSOR, "input_list": [PASSED_TENSOR], "op": ReduceOp.SUM}, "output"),
        (
            "all_to_all",
            {"output_tensor_list": [PASSED_TENSOR], "input_tensor_list": [PASSED_TENSOR]},
            "output_tensor_list",
        ),
        ("barrier", {"device_ids": [0]}, None),
        ("all_gather_object", {"object_list": [PASSED_OBJECT], "obj": PASSED_OBJECT}, "object_list"),
        pytest.param(
            "broadcast_object_list",
            {"object_list": [PASSED_OBJECT], "src": 0},
            "object_list",
            marks=RunIf(max_torch="1.10"),
        ),
        pytest.param(
            "broadcast_object_list",
            {"object_list": [PASSED_OBJECT], "src": 0, "device": torch.device("cpu")},
            "object_list",
            marks=RunIf(min_torch="1.10"),
        ),
        (
            "gather_object",
            {"obj": PASSED_OBJECT, "object_gather_list": [PASSED_OBJECT], "dst": 0},
            "object_gather_list",
        ),
        (
            "scatter_object_list",
            {"scatter_object_output_list": [PASSED_OBJECT], "scatter_object_input_list": [PASSED_OBJECT], "src": 0},
            "scatter_object_output_list",
        ),
        ("monitored_barrier", {"timeout": datetime.timedelta(seconds=1), "wait_all_ranks": False}, None),
    ],
)
@skip_distributed_unavailable
def test_collective_calls_with_created_group(fn_name, kwargs, return_key):
    collective = TorchCollective()
    with mock.patch("torch.distributed.init_process_group"):
        collective.setup()
    with mock.patch("torch.distributed.new_group"):
        collective.create_group()
    fn = getattr(collective, fn_name)
    with mock.patch(f"torch.distributed.{fn_name}", autospec=True) as mock_call:
        result = fn(**kwargs)
    mock_call.assert_called_once_with(**kwargs, group=collective.group)
    if return_key is not None:
        assert result == kwargs[return_key]

    with mock.patch("torch.distributed.destroy_process_group"):
        collective.teardown()


@skip_distributed_unavailable
def test_convert_ops():
    # Test regular names
    assert TorchCollective._convert_to_native_op("band") == ReduceOp.BAND
    assert TorchCollective._convert_to_native_op("bor") == ReduceOp.BOR
    assert TorchCollective._convert_to_native_op("bxor") == ReduceOp.BXOR
    assert TorchCollective._convert_to_native_op("max") == ReduceOp.MAX
    assert TorchCollective._convert_to_native_op("min") == ReduceOp.MIN
    assert TorchCollective._convert_to_native_op("product") == ReduceOp.PRODUCT
    assert TorchCollective._convert_to_native_op("sum") == ReduceOp.SUM
    # Test we are passing through native ops without change
    assert TorchCollective._convert_to_native_op(ReduceOp.BAND) == ReduceOp.BAND
    assert TorchCollective._convert_to_native_op(ReduceOp.BOR) == ReduceOp.BOR
    assert TorchCollective._convert_to_native_op(ReduceOp.BXOR) == ReduceOp.BXOR
    assert TorchCollective._convert_to_native_op(ReduceOp.MAX) == ReduceOp.MAX
    assert TorchCollective._convert_to_native_op(ReduceOp.MIN) == ReduceOp.MIN
    assert TorchCollective._convert_to_native_op(ReduceOp.PRODUCT) == ReduceOp.PRODUCT
    assert TorchCollective._convert_to_native_op(ReduceOp.SUM) == ReduceOp.SUM
    # Test we are handling different casing properly
    assert TorchCollective._convert_to_native_op("BOR") == ReduceOp.BOR
    assert TorchCollective._convert_to_native_op("BoR") == ReduceOp.BOR

    # AVG is very recent!
    if _TORCH_GREATER_EQUAL_1_11:
        assert TorchCollective._convert_to_native_op("avg") == ReduceOp.AVG

    # Test invalid type
    with pytest.raises(ValueError, match="Unsupported op 1 of type int"):
        TorchCollective._convert_to_native_op(1)

    # Test invalid string
    with pytest.raises(ValueError, match="op 'INVALID' is not a member of `Red"):
        TorchCollective._convert_to_native_op("invalid")

    # Test RedOpType
    if _TORCH_GREATER_EQUAL_1_13:
        assert TorchCollective._convert_to_native_op(ReduceOp.RedOpType.AVG) == ReduceOp.RedOpType.AVG
        op = torch.distributed._make_nccl_premul_sum(2.0)  # this returns a ReduceOp
        assert TorchCollective._convert_to_native_op(op) == ReduceOp.PREMUL_SUM
        assert TorchCollective._convert_to_native_op("premul_sum") == ReduceOp.PREMUL_SUM


@skip_distributed_unavailable
@mock.patch.dict(os.environ, {}, clear=True)
def test_repeated_create_and_destroy():
    collective = TorchCollective()
    with mock.patch("torch.distributed.init_process_group"):
        collective.setup(main_address="foo", main_port=123)

    assert not os.environ

    with mock.patch("torch.distributed.new_group") as new_mock:
        collective.create_group()
    new_mock.assert_called_once()

    with pytest.raises(RuntimeError, match="TorchCollective` already owns a group"):
        collective.create_group()

    with mock.patch("torch.distributed.destroy_process_group") as destroy_mock:
        collective.teardown()
    # this would be called twice if `init_process_group` wasn't patched. once for the group and once for the default
    # group
    destroy_mock.assert_called_once()

    assert not os.environ

    with pytest.raises(RuntimeError, match="TorchCollective` does not own a group"):
        collective.teardown()
    destroy_mock.assert_called_once_with(new_mock.return_value)
    assert collective._group is None

    with mock.patch("torch.distributed.new_group"), mock.patch("torch.distributed.destroy_process_group"):
        # check we can create_group again. also chaining
        collective.create_group().teardown()


def collective_launch(fn, device_type, num_devices=1, autosetup_strategy=False, num_groups=1):
    device_to_accelerator = {"cuda": CUDAAccelerator, "cpu": CPUAccelerator}
    accelerator_cls = device_to_accelerator[device_type]
    devices = [torch.device(f"{device_type}:{i}") for i in range(num_devices)]
    strategy = DDPSpawnStrategy(
        accelerator=accelerator_cls(), parallel_devices=devices, cluster_environment=LightningEnvironment()
    )
    launcher = _MultiProcessingLauncher(strategy=strategy)
    collectives = [TorchCollective() for _ in range(num_groups)]
    wrapped = partial(wrap_launch_function, fn, strategy, collectives, devices, autosetup_strategy)
    return launcher.launch(wrapped, strategy, *collectives)


def wrap_launch_function(fn, strategy, collectives, devices, autosetup_strategy, *args, **kwargs):
    with check_destroy_group(autosetup_strategy):  # manually use the fixture for the assertions
        if autosetup_strategy:
            strategy.setup_environment()
        else:
            strategy._set_world_ranks()
            collectives[0].setup(  # only one needs to setup
                world_size=strategy.num_processes,
                main_address=strategy.cluster_environment.main_address,
                backend=strategy._get_process_group_backend(),
                rank=strategy.global_rank,
                timeout=torch_collective.default_pg_timeout,
            )

        fn_params = inspect.signature(fn).parameters
        if "device" in fn_params:
            kwargs["device"] = devices[strategy.global_rank]

        fn(*args, **kwargs)
        torch.distributed.barrier()

        # not necessary since they will be destroyed on process destruction, only added to fulfill the assertions
        for c in collectives:
            c.teardown()

        if autosetup_strategy:
            torch.distributed.destroy_process_group()


def _test_distributed_collectives_fn(strategy, collective, device):
    collective.create_group()

    # all_gather
    tensor_list = [torch.zeros(2, dtype=torch.long, device=device) for _ in range(strategy.num_processes)]
    this = torch.arange(2, dtype=torch.long, device=device) + 2 * strategy.global_rank
    out = collective.all_gather(tensor_list, this)
    expected = torch.arange(2 * strategy.num_processes, device=device).split(2)
    torch.testing.assert_close(tuple(out), expected)

    # reduce
    this = torch.tensor(strategy.global_rank + 1, device=device)
    out = collective.reduce(this, dst=0, op="max")
    expected = torch.tensor(strategy.num_processes, device=device) if strategy.global_rank == 0 else this
    torch.testing.assert_close(out, expected)

    # all_reduce
    this = torch.tensor(strategy.global_rank + 1, device=device)
    out = collective.all_reduce(this, op=ReduceOp.MIN)
    expected = torch.tensor(1, device=device)
    torch.testing.assert_close(out, expected)


@skip_distributed_unavailable
@pytest.mark.parametrize(
    ["device_type", "num_devices", "autosetup_strategy"],
    [
        ("cpu", 1, True),
        ("cpu", 2, True),
        ("cpu", 1, False),
        ("cpu", 2, False),
        pytest.param("cuda", 1, True, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda", 2, True, marks=RunIf(min_cuda_gpus=2)),
        pytest.param("cuda", 1, False, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda", 2, False, marks=RunIf(min_cuda_gpus=2)),
    ],
)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)  # sets CUDA_MODULE_LOADING in torch==1.13
def test_collectives_distributed(device_type, num_devices, autosetup_strategy):
    collective_launch(_test_distributed_collectives_fn, device_type, num_devices, autosetup_strategy)


def _test_distributed_collectives_cuda_fn(strategy, collective):
    collective.create_group()

    this = torch.tensor(1.5, device=strategy.root_device)
    premul_sum = torch.distributed._make_nccl_premul_sum(2.0)
    out = collective.all_reduce(this, op=premul_sum)
    assert out == 3


@skip_distributed_unavailable
@RunIf(min_cuda_gpus=1, min_torch="1.13")
@pytest.mark.parametrize("autosetup_strategy", [True, False])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)  # sets CUDA_MODULE_LOADING in torch==1.13
def test_collectives_distributed_cuda(autosetup_strategy):
    collective_launch(_test_distributed_collectives_cuda_fn, "cuda", 1, autosetup_strategy)


def _test_two_groups(strategy, left_collective, right_collective, device):
    left_collective.create_group(ranks=[0, 1])
    right_collective.create_group(ranks=[1, 2])

    tensor = torch.tensor(strategy.global_rank, device=device)
    if strategy.global_rank in (0, 1):
        tensor = left_collective.all_reduce(tensor)
        assert tensor == 1
    right_collective.barrier()  # avoids deadlock for global rank 1
    if strategy.global_rank in (1, 2):
        tensor = right_collective.all_reduce(tensor)
        assert tensor == 3


@skip_distributed_unavailable
@pytest.mark.parametrize(
    ["device_type", "autosetup_strategy"],
    [
        ("cpu", True),
        ("cpu", False),
        pytest.param("cuda", True, marks=RunIf(min_cuda_gpus=3)),
        pytest.param("cuda", False, marks=RunIf(min_cuda_gpus=3)),
    ],
)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)  # sets CUDA_MODULE_LOADING in torch==1.13
@RunIf(skip_windows=True)
def test_two_groups(device_type, autosetup_strategy):
    collective_launch(_test_two_groups, device_type, 3, autosetup_strategy, num_groups=2)
