# Copyright (c) Meta Platforms, Inc. and affiliates.

import contextlib
import copy
import json
import logging
import queue
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import moodist
import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import cwm.rl.envs.envs
import cwm.text.data
from cwm.common.environment import (
    get_is_rank_zero,
    get_world_size,
    init_torch_distributed,
    set_seed,
    setup_env,
    setup_torch_flags,
)
from cwm.common.params import (
    dataclass_to_dict,
    load_from_cli,
    save_params,
)
from cwm.data.dataset import Dataset
from cwm.fastgen.generate import FastGen
from cwm.fastgen.utils.loading import (
    build_fastgen_model,
    build_tokenizer_from_ckpt,
)
from cwm.logging.logger import (
    add_logger_file_handler,
    initialize_logger,
    set_root_log_level,
)
from cwm.rl.envs import Trajectory, build_env_overrides, get_reward_fn, outcomes
from cwm.rl.envs.api import Env, RewardFn
from cwm.rl.envs.config import TaskIdxDatum, to_task_idx_datum
from cwm.rl.lib.datatypes import DataSource, RolloutInfo
from cwm.rl.lib.dump_traj import (
    create_dump_trajectory_dirs,
    get_rollout_dump_info,
)
from cwm.rl.lib.impgen import ImpGen
from cwm.rl.lib.impgen.api import ImpGenAPI
from cwm.text.datatypes import BaseTextDatum
from cwm.text.tokenizers import Tokenizer, build_tokenizer
from evals.args import RLEvalArgs, build_metrics_logger

Dataset.register_package(cwm.text.data, ["srcs"])

logger = logging.getLogger()


Metrics = dict[str, Any]
Sample = dict[str, Any]


def get_checkpoint_step(
    cfg: RLEvalArgs, pattern: str = r"(?:checkpoint|step)_(?P<step>\d+)"
) -> int:
    global_step = getattr(cfg, "global_step", None)
    if global_step is None:
        path = getattr(cfg, "checkpoint_dir", None)
        if path is None:
            return -1
        match = re.search(pattern, Path(path).name)
        global_step = int(match.group("step")) if match else -1
    return global_step


def load_data(
    dump_samples_path: Path,
    dataset: Dataset[TaskIdxDatum],
    data_queue: moodist.Queue,
    num_rollout_threads: int,
) -> None:
    # For resumed runs, load existing data:
    samples_path = dump_samples_path / "all_metrics.jsonl"
    count_per_source: Counter = Counter()
    # A problem is uniquely identified by the task name and the data source
    if samples_path.exists():
        with samples_path.open("r") as s:
            for line in s:
                sample = json.loads(line)
                count_per_source.update(
                    [
                        (
                            sample["task_name"],
                            DataSource(**sample["data_src"]),
                        )
                    ]
                )
    if count_per_source:
        logger.debug(f"TaskDatum counts: {count_per_source}")

    world_size = get_world_size()
    logger.info("Start data loading")
    for data in dataset:
        assert isinstance(data, TaskIdxDatum)
        assert isinstance(data.src, BaseTextDatum.Source)
        data_src = DataSource.from_base_text_datum_source(data.src)
        task_args = data.val[0]
        needed = task_args.samples_per_prompt or outcomes.infer_n_samples(
            task_args.metrics_spec
        )
        missing = needed - count_per_source[(task_args.name, data_src)]
        for _ in range(missing):
            while data_queue.qsize() > 200:
                time.sleep(0.2)
            data_queue.put_object(data)

    # Tell the rollout threads that the party is over
    for _ in range(num_rollout_threads * world_size):
        data_queue.put_object(None)

    logger.info("Finished data loading")


def rollout(
    args: RLEvalArgs,
    environments_and_rewards: list[tuple[Env, Callable[..., RewardFn]]],
    g: ImpGen,
    data_queue: moodist.Queue,
    done_queue: queue.Queue[bool],
    dump_queue: moodist.Queue,
) -> None:
    while True:
        data = data_queue.get_object()

        if data is None:
            done_queue.put(True)
            dump_queue.put_object(None)
            logger.info("Rollout thread received kill signal")
            return

        assert isinstance(data, TaskIdxDatum)
        logger.info(f"Started work on datum source {data.src}")
        task_args, env_idx, start_args = data.val
        env, rewardfn_ctor = environments_and_rewards[env_idx]
        rewardfn = rewardfn_ctor()

        # Do a rollout
        start = time.monotonic()
        max_attempts = args.max_exceptions + 1
        success = False
        for attempt in range(max_attempts):
            traj = Trajectory()
            try:
                state, tr = env.start(start_args)
                with state:
                    traj.append(tr)
                    # TODO: get max context length and max_gen from the args / env?
                    while not tr.terminal and len(traj.context) < 131072:
                        action = g.generate(
                            tokens=traj.context,
                            max_gen=env.max_action_len(state),
                            temperature=None,
                            stop_str=getattr(env, "stop_str", None),
                        ).tokens
                        logger.debug(f">>> observation:\n{tr.observation_str}")
                        logger.debug(
                            ">>> observation tokens:\n"
                            + " ".join(map(str, tr.observation))
                        )
                        tr = env.step(state, action)
                        tr.add_rewards(rewardfn(tr))
                        logger.debug(f">>> action:\n{tr.action_str}")
                        logger.debug(
                            ">>> action tokens:\n" + " ".join(map(str, tr.action))
                        )
                        assert tr.action == action
                        traj.append(tr)
            except Exception:
                logger.exception(
                    f"Exception during rollout attempt {attempt + 1} / {max_attempts}"
                )
                continue
            else:
                success = True
                break

        if not success:
            logger.error(
                f"Failed to complete rollout for {data.src} after {max_attempts} attempts."
            )
            continue

        end = time.monotonic()
        assert isinstance(data.src, BaseTextDatum.Source)
        data_src = DataSource.from_base_text_datum_source(data.src)

        assert traj.transitions[-1].rewards is not None
        metrics: Metrics = {
            # TODO report return instead?
            "data_src": asdict(data_src),
            "task_name": task_args.name,
            "terminal_rewards": traj.transitions[-1].rewards[-1],
            "terminal_metrics": traj.transitions[-1].outcomes,
        }

        rollout_info = RolloutInfo(
            traj=traj,
            start_args={
                k: v for k, v in start_args.items() if k in args.keep_start_args
            },
            begin_step=-1,
            end_step=-1,
            rl_task_args=args.tasks[env_idx],
            metrics={"rollout/duration": end - start},
            data_src=data_src,
        )

        sample = {"rollouts": [rollout_info], "metrics": [metrics]}

        dump_queue.put_object(sample)

        logger.info(f"Finished work on datum source {data.src}")


def dump_samples(
    dump_queue: moodist.Queue,
    dump_samples_path: Path,
    num_rollout_threads: int,
    dump_mode: str,
    tokenizer: Tokenizer,
) -> None:
    with contextlib.ExitStack() as stack:
        fmetrics = stack.enter_context(
            open(dump_samples_path / "all_metrics.jsonl", "a")
        )

        kill_counts = 0
        open_files: dict[str, Any] = {}
        while True:
            sample = dump_queue.get_object()

            if sample is None:
                kill_counts += 1
                if kill_counts >= num_rollout_threads * get_world_size():
                    logger.info("Dump thread received kill signal")
                    return
                continue

            assert len(sample["rollouts"]) == 1
            assert len(sample["metrics"]) == 1
            task_name = sample["rollouts"][0].rl_task_args.name
            data_src = asdict(sample["rollouts"][0].data_src)
            metrics = {
                "data_src": data_src,
                "task_name": task_name,
                "metrics": sample["metrics"],
            }
            fmetrics.write(json.dumps(metrics) + "\n")
            fmetrics.flush()

            rollout_dump_info = get_rollout_dump_info(
                rollouts=sample["rollouts"],
                metrics=sample["metrics"],
                trajectory_dump_dir=dump_samples_path,
                dump_mode=dump_mode,
                worker_id=None,
                tokenizer=tokenizer,
            )

            if task_name in open_files:
                f = open_files[task_name]
            else:
                file_path = rollout_dump_info.file_path
                f = stack.enter_context(open(file_path, "a"))
                open_files[task_name] = f

            f.write(json.dumps(rollout_dump_info.json) + "\n")
            f.flush()


def thread_wrapper(
    target: Callable, exc_queue: queue.Queue[Exception], *args, **kwargs
) -> None:
    try:
        target(*args, **kwargs)
    except Exception as ex:
        logger.exception(f"Exception in thread {target}:")
        exc_queue.put(ex)


def aggregate_metrics(
    metrics: list[Metrics],
    metrics_spec: dict[str, list[str]],
) -> dict[str, float]:
    """
    Aggregate metrics of a single task.
    """
    # group results by data_src (=prompt) -> metric_name -> list of values
    all_metrics: dict[DataSource, dict] = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        m.pop("task_name")
        data_src = DataSource(**m.pop("data_src"))
        for k, v in m.items():
            all_metrics[data_src][k].append(v)

    # Terminal reward is always aggregated. Other metrics recorded in `outcomes` are aggregated if available.
    count = sum(len(ms["terminal_rewards"]) for ms in all_metrics.values())
    total_terminal_rewards = sum(
        sum(ms["terminal_rewards"]) for ms in all_metrics.values()
    )
    terminal_metrics = [ms["terminal_metrics"] for ms in all_metrics.values()]
    return {
        "terminal_reward_mean": total_terminal_rewards / count,
        "count": count,
        **outcomes.aggregate_outcomes_from_spec(
            metrics_spec,
            terminal_metrics,
        ),
    }


def aggregate_by_task(
    args: RLEvalArgs, dump_samples_path: Path
) -> dict[str, dict[str, float]]:
    metrics_specs_per_task = {t.name: t.metrics_spec for t in args.tasks}

    metrics_path = dump_samples_path / "all_metrics.jsonl"
    metrics_per_task: dict[str, list[Metrics]] = defaultdict(list)
    with metrics_path.open("r") as m:
        for line in m:
            sample = json.loads(line)
            assert len(sample["metrics"]) == 1
            metrics_per_task[sample["task_name"]].append(
                copy.deepcopy(sample["metrics"][0])
            )
    total_metrics = sum(len(metrics) for metrics in metrics_per_task.values())
    logger.info(f"Read {total_metrics} samples from {len(metrics_per_task)} tasks")

    task_results: dict[str, dict[str, float]] = {}

    logger.info("---- Aggregated results ----")
    for task_name, metrics in metrics_per_task.items():
        task_results[task_name] = aggregate_metrics(
            metrics, metrics_specs_per_task[task_name]
        )
        logger.info(f"Task {task_name} results: {task_results[task_name]}")

    return task_results


def setup_mesh(
    args: RLEvalArgs,
) -> tuple[DeviceMesh, torch.distributed.ProcessGroup, torch.distributed.ProcessGroup]:
    """
    Setup the device mesh and process groups for the evals.
    We use a 2D mesh with "dp" and "tp" dimensions, but we actually do data parallel over all ranks.
    Each rank can call g.generate() independently on different prompts, and ImpGen will sync the
    input per tp group under the hood.
    """
    world_size = get_world_size()
    tp_size = args.gen_args.tp_size
    num_tp_groups = world_size // tp_size
    assert (
        num_tp_groups * tp_size == world_size
    ), f"tp_size should divide world_size. ws={world_size}, tps={tp_size}"

    world_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(num_tp_groups, tp_size),
        mesh_dim_names=("dp", "tp"),
    )

    global_rank = world_mesh.get_rank()
    all_group = torch.distributed.new_group(
        world_mesh.mesh.flatten().tolist(), backend="moodist"
    )

    # FastGen creates cuda graphs which requires NCCL process groups,
    # but ImpGen requires moodist because it's using moodist queues,
    # so we create them here
    all_tp_group_ranks = world_mesh.mesh.tolist()
    assert (
        len(all_tp_group_ranks) == num_tp_groups
        and len(all_tp_group_ranks[0]) == tp_size
    ), "each row in the mesh should be a tp group"
    tp_group = None
    for ranks in all_tp_group_ranks:
        pg = torch.distributed.new_group(ranks, backend="moodist")
        if global_rank in ranks:
            tp_group = pg
    assert tp_group is not None

    return world_mesh, all_group, tp_group


def run_rl_evals(
    args: RLEvalArgs,
    global_step: int,
    dump_samples_path: Path,
    world_mesh: DeviceMesh,
    all_group: torch.distributed.ProcessGroup,
    tp_group: torch.distributed.ProcessGroup,
) -> None:
    global_rank = world_mesh.get_rank()

    tokenizer: Tokenizer
    if args.tokenizer is not None:
        tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
    else:
        tokenizer = build_tokenizer_from_ckpt(args.checkpoint_dir)

    g: ImpGenAPI
    fg: FastGen | None = None
    model = build_fastgen_model(
        world_mesh=world_mesh,
        checkpoint_dir=args.checkpoint_dir,
        vocab_parallel=args.gen_args.vocab_parallel,
        loss_parallel=args.gen_args.loss_parallel,
    )

    fg = FastGen(
        args.gen_args,
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        tp_mesh=world_mesh["tp"],
    )

    torch.cuda.empty_cache()

    g = ImpGen(
        fg,
        tp_group.rank(),
        tp_group,
    )

    runtime_kwargs = {"tokenizer": g.tokenizer}
    exc_queue = queue.Queue[Exception]()

    # Instantiate one env per task, in order.
    # Even though multiple tasks might use the same env, since an env interface is a read-only instance this is not too costly.
    environments_and_rewards = [
        (build_env_overrides(task, **runtime_kwargs), get_reward_fn(task.reward_fn))
        for task in args.tasks
    ]

    data_queue = moodist.Queue(all_group, location=0)
    dump_queue = moodist.Queue(all_group, location=0)
    done_queue = queue.Queue[bool]()

    loading_thread = None
    if global_rank == 0:
        # we load the data on rank 0 and distribute it to all ranks

        # Build the required dataloaders
        # NOTE: we partition by global rank after chaining, so all ranks read every line of every file (returning only those assigned to them)
        # This isn't super efficient but for the typically small evals datasets it should be fine.
        # As an alternative, we could split the eval tasks by rank
        task_datasets = {}
        for env_idx, task in enumerate(args.tasks):
            assert (
                task.path is not None
            ), "For now at least, eval tasks must have a path"
            logger.info(f"Creating dataloader for task {task.name}")
            dataset_path = Path(task.path)
            to_task_idx_datum_te = partial(to_task_idx_datum, task, env_idx)
            task_datasets[task.name] = Dataset.from_jsonl(dataset_path).map(
                to_task_idx_datum_te
            )

        logger.info("Creating chained loader")
        dataset = Dataset.chain(list(task_datasets.values()))

        # Start the data loading thread that loads prompts and puts them in data_queue
        loading_thread = threading.Thread(
            target=thread_wrapper,
            kwargs=dict(
                target=load_data,
                dump_samples_path=dump_samples_path,
                dataset=dataset,
                data_queue=data_queue,
                num_rollout_threads=args.num_rollout_threads,
                exc_queue=exc_queue,
            ),
        )
        loading_thread.start()
        logger.debug(f"Data thread started on global_rank = {global_rank}")

    # Start the rollout threads that read from the data queue, do a rollout, and push the result to a trajectory queue
    rollout_threads = []
    for _ in range(args.num_rollout_threads):
        t = threading.Thread(
            target=thread_wrapper,
            kwargs=dict(
                target=rollout,
                args=args,
                environments_and_rewards=environments_and_rewards,
                g=g,
                data_queue=data_queue,
                done_queue=done_queue,
                dump_queue=dump_queue,
                exc_queue=exc_queue,
            ),
        )
        rollout_threads.append(t)
        t.start()

    # Start the thread that dumps samples to disk on rank 0
    dump_thread = None
    if global_rank == 0:
        dump_thread = threading.Thread(
            target=thread_wrapper,
            kwargs=dict(
                target=dump_samples,
                dump_queue=dump_queue,
                dump_samples_path=dump_samples_path,
                num_rollout_threads=args.num_rollout_threads,
                dump_mode=args.dump_mode,
                tokenizer=tokenizer,
                exc_queue=exc_queue,
            ),
        )
        dump_thread.start()

    logging.info("Start generating")
    done = False
    perf_log_freq = args.perf_log_freq
    last_log_time = time.time()
    ncalls = 0
    nseqs = 0
    ntoks = 0
    while not done:
        done = g.work()
        curr_time = time.time()
        time_elapsed = curr_time - last_log_time
        if time_elapsed > perf_log_freq:
            calls_made = g.i - ncalls
            seqs_generated = g.nseqs - nseqs
            toks_generated = g.ntoks - ntoks
            ncalls = g.i
            nseqs = g.nseqs
            ntoks = g.ntoks

            calls_per_sec = calls_made / time_elapsed
            seqs_per_sec = seqs_generated / time_elapsed
            toks_per_sec = toks_generated / time_elapsed

            last_log_time = curr_time
            logger.info(
                f"Generation performance: {calls_per_sec} calls/s, {seqs_per_sec} seqs/s, {toks_per_sec} toks/s"
            )

        if done_queue.qsize() >= args.num_rollout_threads:
            # Note: we don't break the loop here because some generations are still in progress
            g.stop()

        try:
            ex = exc_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            raise RuntimeError("Exception in thread") from ex

    logging.info("Generation done")

    # Close all threads
    if loading_thread:
        loading_thread.join()
        logging.info("Joined loading_thread")
    if rollout_threads:
        for t in rollout_threads:
            t.join()
        logging.info("Joined all rollout_threads")
    if fg is not None:
        logger.info("Destroying fg")
        fg.destroy()  # type: ignore
    if dump_thread:
        dump_thread.join()
        logging.info("Joined dump_thread")
        if args.run_metrics_aggregation:
            task_results = aggregate_by_task(args, dump_samples_path)
            metrics = {}
            for task_name, results in task_results.items():
                for k, v in results.items():
                    metrics[f"{task_name}/{k}"] = v
            assert args.metric_log_dir is not None
            with build_metrics_logger(
                Path(args.metric_log_dir), "rl_eval"
            ) as eval_logger:
                eval_logger.log_metrics(metrics, step=global_step)


def eval_model_from_checkpoint(args: RLEvalArgs) -> None:
    # Validate samples_per_prompt and at_k args
    if args.run_metrics_aggregation:
        for task_args in args.tasks:
            outcomes.validate_aggregation_spec(
                task_args.metrics_spec,
                task_args.samples_per_prompt
                or outcomes.infer_n_samples(task_args.metrics_spec),
            )

    assert args.dump_dir is not None
    dump_path = Path(args.dump_dir)
    dump_samples_path = dump_path / "trajectories"
    if get_is_rank_zero():
        dump_path.mkdir(parents=True, exist_ok=True)
        create_dump_trajectory_dirs(dump_samples_path, args.tasks)
        save_params(args, dump_path / "rl_eval_config.yaml")
    add_logger_file_handler(dump_path / "rl_eval.log")

    global_step = get_checkpoint_step(args)
    logger.info(f"Global step: {global_step}")

    setup_env(mp_spawn_method=args.setup.spawn_method)
    init_torch_distributed(timeout=args.setup.torch_init_timeout)

    setup_torch_flags(**dataclass_to_dict(args.setup))
    set_seed(args.seed)

    world_mesh, all_group, tp_group = setup_mesh(args)

    # Queue to send a shutdown signal at the end
    # (otherwise workers will quit the process group prematurely, taking down the run)
    shutdown_queue = moodist.Queue(all_group, location=0)

    run_rl_evals(args, global_step, dump_samples_path, world_mesh, all_group, tp_group)
    if get_is_rank_zero():
        logger.info("Shutting down all ranks")
        for _ in range(get_world_size()):
            shutdown_queue.put_object(None)

    logger.info("Waiting for shutdown signal")
    shutdown_queue.get_object()
    logger.info("Received shutdown signal, leaving.")

    torch.distributed.barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    initialize_logger()
    eval_args = load_from_cli(RLEvalArgs, from_config_file=True, with_preset=True)
    set_root_log_level(eval_args.log_level)
    eval_model_from_checkpoint(eval_args)


if __name__ == "__main__":
    main()
