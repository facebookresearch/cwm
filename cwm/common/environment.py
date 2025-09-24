# Copyright (c) Meta Platforms, Inc. and affiliates.

import atexit
import logging
import multiprocessing as mp
import os
import random
import shutil
import socket
import subprocess
import tempfile
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import moodist
import numpy as np
import torch

logger = logging.getLogger()


@lru_cache
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache
def get_slurm_job_id() -> int:
    return int(os.environ.get("SLURM_JOB_ID", 0))


@lru_cache
def get_slurm_job_name() -> str:
    return os.environ.get("SLURM_JOB_NAME", "")


@lru_cache
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    if get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    return 0


@lru_cache
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    if get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    return 0


@lru_cache
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    if get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    return 1


@lru_cache
def get_is_rank_zero() -> bool:
    return get_global_rank() == 0


@lru_cache
def get_master_port() -> int:
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
    rng = random.Random(int(os.environ.get("SLURM_JOB_ID", -1)))
    return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


@lru_cache
def get_master_addr() -> str:
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    if get_is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        )
        return hostnames.split()[0].decode("utf-8")
    return "127.0.0.1"


def setup_env(
    env_vars: dict | None = None, *, mp_spawn_method: str | None = None
) -> None:
    env_vars = env_vars or {}

    # When using Triton, it attempts to locate prebuilt kernels in a cache
    # located at ~/.triton/cache, but when that's backed by NFS this can fail
    # with a "OSError: [Errno 116] Stale file handle" error. If we were to set
    # it to a local directory it would belong to the first user who created it
    # and it would fail for the job of any other successive user assigned to
    # that machine. To avoid all this mess we use a temporary per-process cache.
    triton_cache_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    env_vars["TRITON_CACHE_DIR"] = triton_cache_dir

    # We change the tmp dir to /scratch in case it's slurm job
    # This avoids filling up the host's usually limited tmpfs
    # A full tmpfs leads to very slow creation of processes and weird bugs
    if get_is_slurm_job():
        new_tmp = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if new_tmp.exists():
            env_vars["TMP_DIR"] = str(new_tmp)
            env_vars["TMPDIR"] = str(new_tmp)

    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            logger.warning(f"Setting env variable {name} to {value}")

    mp.set_start_method(mp_spawn_method)
    with mp.Manager():
        pass


def init_torch_distributed(
    backend: str = "nccl", *, timeout: int | None = None
) -> None:
    local_rank = get_local_rank()

    if get_is_torch_run():
        logger.info(f"Run launched with torchrun, local rank: {local_rank}")
    elif get_is_slurm_job():
        logger.info(f"Run launched with slurm, local rank: {local_rank}")
    else:
        logger.info("Single GPU job")

    logger.info(
        f"This process is rank {get_global_rank()} out of {get_world_size()}, "
        f"running on node {socket.gethostname()} as process {os.getpid()}"
    )

    logger.info(f"ENV: {os.environ}")

    nccl_options = torch.distributed.ProcessGroupNCCL.Options()

    # To efficiently utilize a GPU, the compute kernels should provide enough
    # work to completely fill the GPU. The comms kernels, on the other hand, are
    # bottlenecked on some external resource (e.g., network bandwidth) thus they
    # don't benefit from more GPU resources and only require a fraction of it.
    # If compute goes first, comms can only start once the compute is finished,
    # and at that point it will leave most of the GPU idle. However, if comms
    # goes first, compute can overlap with it.
    # To ensure this happens we must tell CUDA to prioritize comms kernels.
    nccl_options.is_high_priority_stream = True

    # NCCL can segfault in some specific circumstances, such as when using bound
    # device ids, non-blocking init, communicator splits and multiple threads (such
    # as PyTorch's autograd thread). A workaround is to disable non-blocking init.
    # See https://github.com/NVIDIA/nccl/issues/1605
    nccl_options.config.blocking = 1

    if backend == "moodist":
        moodist.enable_cpu_allocator()
        moodist.enable_cuda_allocator()
        moodist.set_prefer_kernel_less(True)

    scheme = "moodist" if hasattr(moodist, "TcpStore") else "tcp"

    logger.info(f"PyTorch rendezvous using scheme {scheme}")

    store, _, _ = next(
        torch.distributed.distributed_c10d.rendezvous(
            url=f"{scheme}://{get_master_addr()}:{get_master_port()}",
            rank=get_global_rank(),
            world_size=get_world_size(),
            timeout=timedelta(seconds=timeout) if timeout else timedelta(minutes=5),
        )
    )

    logger.info("PyTorch established all connections for rendezvous via Store")

    # We can now set a different timeout for individual operations on the Store
    if timeout is not None:
        store.set_timeout(timedelta(seconds=timeout))

    # set GPU device
    assert 0 <= local_rank < torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=f"cuda:{backend},cpu:gloo",
        rank=get_global_rank(),
        world_size=get_world_size(),
        store=torch.distributed.PrefixStore("default_pg", store),
        timeout=timedelta(seconds=timeout) if timeout else None,
        pg_options=nccl_options,
        # Enables eager background init and allows other PGs (DP, TP, ...)
        # to be "split" from the global PG, thus accelerating job start-up.
        # FIXME Currently disabled to workaround https://github.com/pytorch/pytorch/issues/153960
        # device_id=torch.device("cuda", local_rank),
    )

    logger.info("PyTorch completed preliminary initialization of comms")

    torch.distributed.barrier()

    logger.info(
        "PyTorch completed background initialization of comms, which are now usable"
    )


def set_seed(seed: int = 0, *, deterministic: bool = False) -> None:
    # Note: setting deterministic algorithms can make some PyTorch
    # operators slower. Only use when we absolutely need bitwise
    # reproducibility (eg CI...)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_torch_flags(
    *,
    cuda_matmul_allow_tf32: bool = True,
    cuda_allow_bf16_reduced_precision_reduction: bool = True,
    autograd_detect_anomaly: bool = False,
    **kwargs,
) -> None:
    if cuda_matmul_allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.warning(
            "WARNING: Setting torch.backends.matmul.allow_tf32 to True. This is faster but less accurate.",
        )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        cuda_allow_bf16_reduced_precision_reduction
    )
    torch.autograd.set_detect_anomaly(autograd_detect_anomaly)
    torch._C._set_print_stack_traces_on_fatal_signal(True)
