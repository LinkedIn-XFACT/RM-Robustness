# Taken and modified from https://github.com/mnoukhov/async_rlhf
# Copyright 2024 The AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
import vllm
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_world_group,
    init_model_parallel_group,
)
from vllm.executor.gpu_executor import GPUExecutor


def custom_initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.
    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    world_size: int = 1  # SingleGPULLM logic: only use a single GPU
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    # global _TP
    assert vllm.distributed.parallel_state._TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    vllm.distributed.parallel_state._TP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True
    )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    # global _PP
    assert vllm.distributed.parallel_state._PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    vllm.distributed.parallel_state._PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False
    )


def init_world_group(ranks: List[int], local_rank: int, backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[[0]],  # SingleGPULLM logic: only use a single GPU
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=False,
        use_tpu_communicator=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False
    )


def _init_executor(self) -> None:
    """Initialize the worker and load the model."""
    assert self.parallel_config.world_size == 1, "GPUExecutor only supports single GPU."

    self.driver_worker = self._create_worker(local_rank=self.device_config.device.index)
    self.driver_worker.init_device()
    self.driver_worker.load_model()


# monkey patch the function
def vllm_single_gpu_patch():
    vllm.distributed.parallel_state.init_world_group = init_world_group
    vllm.distributed.parallel_state.initialize_model_parallel = custom_initialize_model_parallel
    GPUExecutor._init_executor = _init_executor