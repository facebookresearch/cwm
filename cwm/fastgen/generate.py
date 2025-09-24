# Copyright (c) Meta Platforms, Inc. and affiliates.

import bisect
import os
import queue
import signal
from collections import deque
from collections.abc import Generator, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from typing import (
    Any,
    Literal,
)

import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.functional import cross_entropy
from torch.profiler import ProfilerActivity, profile

from cwm.text import tokenizers

from . import api
from . import forward as fwd
from .utils import metrics, sampling, tune
from .utils.cache import Cache, DynCache, DynCacheLane, RawCache
from .utils.collectives import Collectives
from .utils.iset import ISet
from .utils.misc import hashints

logger = getLogger()


@dataclass
class GenArgs:
    # Default sampling arguments
    use_sampling: bool = True
    temperature: float = 0.7
    top_p: float = 1.0

    logprobs: bool = False
    prompt_logprobs: bool = False

    max_batch: int = 128
    max_seq: int = -1  # negative for auto tune
    max_gen: int | None = None

    host_cache_gb: float = 0  # 0 to disable
    num_cuda_graphs: int = 16
    cache_block: int = 256
    # prefill bounds, in tokens
    max_prefill: int = 4096  # negative for auto tune
    min_prefill: int = 512  # TODO: tune this default
    min_prefill_batch: int = 8

    prefill_gb: float = 2
    gpu_gb: float = 74

    # frequency, in decoding iterations, at which
    # generation lanes are checked for termination
    sync_freq: int = 10
    # if ``handover_frequency`` > 0, hands over control
    # to the calling code periodically by yielding ``None``
    # every ``handover_frequency`` decoding iterations
    handover_frequency: int = 0


@dataclass
class DecodeGraph:
    state: fwd.ModelState
    graph: torch.cuda.CUDAGraph | None
    logits: torch.Tensor | None


@dataclass
class Lane(api.Packet):
    maxlen: int = 0
    prompt: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    blockset: ISet = field(default_factory=ISet)

    @staticmethod
    def from_pkt(
        p: api.Packet,
        max_seq: int,
        max_gen: int | None,
    ) -> "Lane":
        max_gen = p.max_gen if p.max_gen is not None else max_gen
        if max_gen is None:
            maxlen = max_seq
        else:
            maxlen = len(p.tokens) + max_gen
            maxlen = min(maxlen, max_seq)
        return Lane(
            thread_id=p.thread_id,
            temperature=p.temperature,
            top_p=p.top_p,
            max_gen=max_gen,
            prompt=p.tokens,
            maxlen=maxlen,
            stop_str=p.stop_str,
        )

    def all_tokens(self) -> list[int]:
        "The prompt and generated tokens."
        return self.prompt + self.tokens

    def add_block(self, bid: int) -> None:
        self.blockset.add(bid)
        self.blocks.append(bid)

    def free(self, free_set: ISet) -> None:
        free_set |= self.blockset
        self.blockset.clear()
        self.blocks = []

    def set_blocks(self, bset: ISet) -> None:
        assert not self.blockset
        self.blockset = bset
        self.blocks = bset.tolist()

    def __len__(self) -> int:
        "Length in tokens, including the prompt."
        return len(self.prompt) + len(self.tokens)


class FastGen(api.GenAPI[api.NoErr]):
    def __init__(
        self,
        args: GenArgs,
        model: torch.nn.Module,
        tokenizer: tokenizers.Tokenizer,
        dtype: torch.dtype,
        device: torch.device,
        tp_mesh: DeviceMesh | None,
        profile_sig: int | None = signal.SIGUSR2,
    ):
        self.device = device
        tp_mesh = tp_mesh if tp_mesh and tp_mesh.size() > 1 else None
        self.tp_mesh = tp_mesh
        self.tp_coll = Collectives(tp_mesh) if tp_mesh else None
        self.generating = False
        self.last_used_gb: float | None = None

        self.profile_state: Literal[
            "inactive",
            "requested",
            "active",
        ] = "inactive"
        self.profile_iter = 0
        self.profile_iters = int(os.environ.get("FG_PROFILE_ITERS", "8"))
        self.profile_dir = os.environ.get("FG_PROFILE_DIR", "/tmp")
        self.profile_ctx: Any = None
        self.profile_obj: Any = None

        if profile_sig is not None:
            if signal.getsignal(profile_sig) is not signal.SIG_DFL:
                logger.warning(
                    f"Signal handler for {profile_sig} is set;"
                    " Fastgen profile handler was not installed"
                )
            else:
                signal.signal(profile_sig, self._profile_sig_handler)
                logger.info(
                    f"Profiling signal handler set for signal {profile_sig}",
                )

        if tp_mesh:
            r = int(torch.rand(1)[0] * 0x7FFFFFFF)
            self._check_consistent(r, "random seed")

        logger.info("Initializing generator")
        self._log_cuda_mem()

        prefill_gb = max(args.prefill_gb, 1)
        mem_params = tune.mem_params(tp_mesh, model, prefill_gb, args.gpu_gb)

        self.cache_shard = (0, 1)
        # page_len is the logical page size while
        # block_len is the physical block length
        # and local to each TP rank
        if getattr(model, "kv_lora_rank", 0) > 0:
            # FlashMLA only supports blocks of size 64
            self.page_len = 64
            self.block_len = self.page_len
            if tp_mesh:
                tp_size = tp_mesh.size()
                tp_rank = tp_mesh.get_local_rank()
                self.cache_shard = (tp_rank, tp_size)
                self.page_len *= tp_size
        else:
            self.page_len = args.cache_block
            self.block_len = self.page_len
        mem_params.round_to(self.page_len)
        self.gen_args = args
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = list(tokenizer.stop_tokens)
        self.max_batch = args.max_batch
        self.min_prefill_batch = args.min_prefill_batch
        if self.min_prefill_batch >= self.max_batch:
            self.min_prefill_batch = 1
        self.max_prefill = args.max_prefill
        if self.max_prefill < 0:
            self.max_prefill = mem_params.prefill_tokens
        self.max_seq = args.max_seq
        if self.max_seq <= 0:
            tp_size = tp_mesh.size() if tp_mesh else 1
            # defaults for some models on 80GB GPUs
            match model.n_layers:
                case 16:
                    # llama3 1.5B
                    self.max_seq = tp_size * 38400
                case 32:
                    # llama3 8B
                    self.max_seq = tp_size * 384000
                case 64:
                    # cwm 32B baseline
                    if tp_size >= 8:
                        self.max_seq = tp_size * 256000
                    elif tp_size >= 4:
                        self.max_seq = tp_size * 230400
                    else:
                        self.max_seq = tp_size * 166400
                case 80:
                    # llama3 70B
                    self.max_seq = tp_size * 122880
                case _:
                    logger.info(
                        "Using auto tuned max_seq because the model type is unknown"
                    )
                    self.max_seq = mem_params.cache_tokens
        logger.info(f"Setting max_prefill={self.max_prefill} max_seq={self.max_seq}")
        self.gen_logprobs = args.logprobs or args.prompt_logprobs
        self.gen_prompt_logprobs = args.prompt_logprobs

        assert self.max_seq > 0
        assert self.max_seq % self.page_len == 0
        assert self.max_seq >= self.max_batch * self.page_len
        assert args.sync_freq < self.page_len

        nblocks = self.max_seq // self.page_len
        self.cache = RawCache.build(
            model=model,
            length=nblocks * self.block_len,
            dtype=dtype,
            device=device,
        )
        _, cache_shards = self.cache_shard
        bytes_per_tok = self.cache.bytes_per_token() / cache_shards
        self.cache_ready: torch.cuda.Event | None = None
        self.host_cache = Cache(
            limit_toks=int(args.host_cache_gb * 1e9 / bytes_per_tok),
            page_len=self.page_len,
            node_len=self.block_len,
        )
        logger.info("Cache bytes per token: %s", bytes_per_tok)
        logger.info("Host cache node count limit: %s", self.host_cache.limit)
        self.host_cache.preallocate(gpu_cache=self.cache)

        logger.info("Allocated kv cache and host cache")
        self._log_cuda_mem()

        self.parking = deque[Lane]()

        # decoder device tensors; will be copied
        # in the model state before decode calls
        self.nactive = 0
        self.tokens = torch.randint(
            low=0,
            high=model.vocab_size,
            size=(self.max_batch,),
            dtype=torch.int,
            device=device,
        )
        self.seqlen = torch.zeros(
            self.max_batch,
            dtype=torch.int,
            device=device,
        )
        self.maxlen = torch.zeros(
            self.max_batch,
            dtype=torch.int,
            device=device,
        )
        self.temps = torch.zeros(
            self.max_batch,
            dtype=torch.float,
            device=device,
        )
        self.top_ps = torch.zeros(
            self.max_batch,
            dtype=torch.float,
            device=device,
        )
        self.blktbl = torch.zeros(
            (self.max_batch, nblocks),
            dtype=torch.int,
            device=device,
        )

        self.free_blocks = ISet.interval(0, nblocks)
        self.nblocks = nblocks

        # per-lane host data
        self.lane = [Lane("dead") for _ in range(self.max_batch)]

        self.top_p_lanes = set[int]()

        self.use_graphs = not bool(os.environ.get("FG_NO_CUDA_GRAPHS"))
        if not self.use_graphs:
            logger.warning("FastGen is not using graphs")

        # decide the graph batch sizes
        num_graphs = max(2, args.num_cuda_graphs)
        batch_sizes: list[int] = sorted(
            set(
                torch.linspace(
                    start=1,
                    end=self.max_batch,
                    steps=num_graphs,
                    dtype=torch.float32,
                    device="cpu",
                )
                .round()
                .int()
                .tolist()
            )
        )
        self.graph_batch_sizes = batch_sizes

        model.rope_freqs = fwd.rope_freqs(model).to(device)
        self.update_model()

        # create the graphs
        graphs: list[DecodeGraph] = []
        pool = None
        for bs in reversed(batch_sizes):
            state = fwd.ModelState(
                bs,
                self.tokens,
                self.blktbl,
                self.block_len,
                self.cache,
                self.cache_shard,
                device,
            )

            if self.use_graphs:
                with (
                    self.tp_coll.capture()
                    if self.tp_coll is not None
                    else nullcontext()
                ):
                    # let triton compile the kernels
                    fwd.decode(self.model, self.tp_coll, state)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        logits = fwd.decode(self.model, self.tp_coll, state)
                    if pool is None:
                        pool = graph.pool()

            else:
                graph = None
                logits = None

            graphs.append(DecodeGraph(state, graph, logits))

        self.graphs = list(reversed(graphs))

        if self.use_graphs:
            logger.info("Created %s decoding graphs", len(self.graphs))
            self._log_cuda_mem()

        # clear any changes that may have happened
        # during graph recording
        self.cache.clear()

        logger.info("Done initializing")

    def update_model(self) -> None:
        fwd.update_model(self.model, self.tp_coll)

    def destroy(self) -> None:
        """
        Free some internal data; it was observed to
        be necessary to avoid hangs during
        ``torch.distributed.destroy_process_group()``
        """
        while self.graphs:
            self.graphs.pop()

    def request_profile(self) -> None:
        """
        Request the generation of a profile trace.
        """
        if self.profile_state == "inactive":
            logger.info("Requesting profiling trace...")
            self.profile_state = "requested"

    def _profile_sig_handler(self, _signum: int, _: Any) -> None:
        self.request_profile()

    def _profile_step(self) -> None:
        if self.profile_state == "active":
            self.profile_iter += 1
            if self.profile_iter == self.profile_iters:
                self._profile_done()
            return

        assert self.profile_state == "requested"
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.profile_iter = 0
        self.profile_ctx = profile(activities=activities, with_stack=True)
        self.profile_obj = self.profile_ctx.__enter__()
        self.profile_state = "active"

    def _profile_done(self) -> None:
        if self.profile_state != "active":
            return
        self.profile_state = "inactive"
        try:
            self.profile_ctx.__exit__(None, None, None)
            tstamp = datetime.now().strftime("%y%m%d-%H%M%S")  # noqa: DTZ005
            pid = os.getpid()
            dumpf = f"{self.profile_dir}/fastgen_{tstamp}_{pid}.json.gz"
            self.profile_obj.export_chrome_trace(dumpf)
            logger.info("Profile chrome trace exported to %s", dumpf)
        except Exception as e:
            logger.warning(f"Error while exporting profile trace: {e!r}")

    def generate(
        self,
        q: queue.Queue[api.Packet | None],
    ) -> Generator[api.Packet | None, None, None]:
        assert not self.generating, "multiple generators cannot run concurrently"
        self.generating = True
        yield from self._generate(q, self.gen_args.handover_frequency)  # type: ignore
        self._profile_done()
        self.generating = False

    @torch.inference_mode()
    def _generate(
        self,
        q: queue.Queue[api.Packet | None],
        handover_frequency: int = 0,
    ) -> Generator[api.Packet | None, None, None]:
        eoq = False
        last_nactive = 0
        num_iters = 0
        while not eoq or self.parking or self.nactive > 0:
            if handover_frequency > 0:
                num_iters += 1
                if num_iters == handover_frequency:
                    num_iters = 0
                    yield None
            if self.profile_state != "inactive":
                self._profile_step()

            done = []
            tokens_list = []
            logprobs_list = []
            sample_logprobs_list = []

            self._prepare_lanes()

            if last_nactive != self.nactive:
                last_nactive = self.nactive
                metrics.dump(
                    "nactive",
                    lanes=self.nactive,
                    free_blocks=len(self.free_blocks),
                    nblocks=self.nblocks,
                )

            if self.nactive > 0:
                tokens, logprobs, sample_logprobs = self._decode()
                toolong = self.seqlen >= self.maxlen
                toolong = toolong[: self.nactive]
                done += self._ended_lanes(tokens, toolong)
                tokens_list += tokens.tolist()
                logprobs_list += logprobs
                sample_logprobs_list += sample_logprobs

            self._sync_host_cache()
            self.host_cache.maintain()

            eoq, added = self._add_lanes(eoq, q)
            if added:
                tokens, logprobs, sample_logprobs, toolong = self._prefill(added)
                done += self._ended_lanes(tokens, toolong)
                tokens_list += tokens.tolist()
                logprobs_list += logprobs
                sample_logprobs_list += sample_logprobs

            # iterate in reverse so we can kill multiple
            # lanes; see _kill_lane()
            for i in reversed(range(self.nactive)):
                self.lane[i].tokens += tokens_list[i]
                if self.gen_logprobs:
                    lp = self.lane[i].logprobs or []
                    lp += logprobs_list[i]
                    self.lane[i].logprobs = lp
                    slp = self.lane[i].sample_logprobs or []
                    slp += sample_logprobs_list[i]
                    self.lane[i].sample_logprobs = slp
                if done[i] or self._has_stop_str(self.lane[i], len(tokens_list[i])):
                    pkt = self._trim_packet(self.lane[i])
                    metrics.dump("yield", tokens=len(pkt.tokens))
                    yield pkt
                    # _kill_lane() will shuffle lanes and may
                    # clobber pending copies to host memory;
                    # wait for coherence before continuing
                    self._sync_host_cache()
                    self._kill_lane(i)

    def _has_stop_str(self, lane: Lane, n_new: int) -> bool:
        if lane.stop_str is None:
            return False
        # we don't know for sure how many tokens the stop_str
        # will span, but an upper bound is len(stop_str)
        token_span = len(lane.stop_str) + n_new
        tokens = lane.tokens[-token_span:]
        return lane.stop_str in self.tokenizer.decode(tokens)

    def _prepare_lanes(self) -> None:
        """
        Prepare the lanes to make sure they have enough
        blocks allocated for a decoding call. During
        preparation, some active lanes may be parked
        to free cache blocks.
        """
        sync_freq = self.gen_args.sync_freq
        page_len = self.page_len

        # evict short lanes first
        lanes = self.lane[: self.nactive]
        ordered = sorted((-len(ln), i) for i, ln in enumerate(lanes))
        decr = [i for _, i in ordered]

        idx = 0
        while idx < self.nactive:
            lane = self.lane[decr[idx]]

            avail = len(lane.blocks) * page_len
            if len(lane) + sync_freq <= avail:
                idx += 1
                continue
            # else, we need one more block

            if not self.free_blocks:
                last = self.nactive - 1
                if last > 0:
                    self._kill_lane(decr[last], park=True)
                    if idx == last:
                        break
                    # fixup decr[] in case we still
                    # have to visit the lane that moved
                    for nxt in range(idx, last):
                        if decr[nxt] == last:
                            decr[nxt] = decr[last]
                            break

            if self.free_blocks:
                bid = self.free_blocks.popleft()
                self.blktbl[decr[idx], len(lane.blocks)] = bid
                lane.add_block(bid)
            else:
                # it is possible that there are no free blocks;
                # but in this case nactive must be 1 and we're
                # going to hit maxlen pretty soon
                assert self.nactive == 1
                idx = self.nactive

            avail = len(lane.blocks) * page_len
            assert min(self.max_seq, len(lane) + sync_freq) <= avail

    def _add_lanes(
        self,
        eoq: bool,
        q: queue.Queue[api.Packet | None],
    ) -> tuple[bool, int]:
        """
        Add new lanes to the ``self.lane`` list.

        Returns:
            tuple[bool, int]:
                A boolean indicating if the queue has run
                out of packets and the number of lanes
                added.
        """
        buffer = 128  # in number of tokens
        addend = buffer + self.page_len - 1

        idx = self.nactive
        tokens = 0
        cksum = 0

        while idx < self.max_batch:
            lane: Lane | None = None
            if parked := bool(self.parking):
                lane = self.parking.popleft()
            elif not eoq:
                try:
                    pkt = q.get(block=(idx == 0))
                except queue.Empty:
                    break
                if pkt is None:
                    eoq = True
                    cksum = hashints([cksum, 0])
                else:
                    pkth = hashints(pkt.tokens)
                    cksum = hashints([cksum, pkth])
                    lane = Lane.from_pkt(
                        pkt,
                        self.max_seq,
                        self.gen_args.max_gen,
                    )

            if lane is None:
                break

            avail = len(self.free_blocks)
            ask = (len(lane) + addend) // self.page_len
            if ask > avail:
                if parked:
                    self.parking.appendleft(lane)
                else:
                    self.parking.append(lane)
                break

            if parked:
                metrics.dump("unpark", tokens=len(lane))

            blocks, self.free_blocks = self.free_blocks.take(ask)
            lane.set_blocks(blocks)
            self.lane[idx] = lane
            tokens += len(lane)
            idx += 1

        if tokens < self.gen_args.min_prefill:
            if idx - self.nactive < self.min_prefill_batch:
                if not eoq and self.nactive > 0:
                    # wait some more before prefilling
                    while idx > self.nactive:
                        idx -= 1
                        lane = self.lane[idx]
                        lane.free(self.free_blocks)
                        self.parking.appendleft(lane)

        if cksum:
            self._check_consistent(cksum, "input queue")

        return eoq, idx - self.nactive

    def _check_consistent(self, val: int, what: str) -> None:
        "The ``val`` argument must be a representable int32."
        min_val = self.tensor([val])
        if self.tp_mesh:
            torch.distributed.all_reduce(
                min_val,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_mesh.get_group(),
            )
        if int(min_val[0]) != val:
            raise RuntimeError(f"Inconsistent {what} across mp ranks")

    def _sync_host_cache(self) -> None:
        """
        Ensure that the host cache is coherent before
        returning.
        """
        if self.cache_ready is not None:
            self.cache_ready.wait()
            self.cache_ready = None

    def _ended_lanes(
        self,
        tokens: torch.Tensor,
        toolong: torch.Tensor,
    ) -> list[int]:
        has_eos = torch.zeros_like(toolong)
        for st in self.stop_tokens:
            has_eos = torch.logical_or(has_eos, (tokens == st).sum(1))
        return (has_eos | toolong).tolist()

    def _trim_packet(self, pkt: Lane) -> api.Packet:
        """
        Trim an answer in place so that it is cut after
        the eos token. The input packet is returned.
        """
        trim = len(pkt.tokens)
        n = max(0, len(pkt.tokens) - self.gen_args.sync_freq - 1)
        for st in self.stop_tokens:
            if st in pkt.tokens[n:]:
                trim = min(trim, pkt.tokens.index(st, n) + 1)
        if pkt.max_gen is not None:
            trim = min(trim, pkt.max_gen)
        if (stopstr := pkt.stop_str) is not None:
            decode = self.tokenizer.decode
            if stopstr in decode(pkt.tokens[:trim]):
                # f is a trim pos s.t. stopstr not in tokens[:f]
                # t is a trim pos s.t. stopstr in tokens[:t]
                f, t = 0, trim
                while t - f > 1:
                    m = f + (t - f) // 2
                    if stopstr in decode(pkt.tokens[:m]):
                        t = m
                    else:
                        f = m
                assert f == t - 1
                trim = t

        pkt.tokens = pkt.tokens[:trim]
        if pkt.sample_logprobs is not None:
            pkt.sample_logprobs = pkt.sample_logprobs[:trim]
        if pkt.logprobs is not None:
            if self.gen_prompt_logprobs:
                nprompt = len(pkt.prompt) - 1
                pkt.logprobs = pkt.logprobs[: nprompt + trim]
            else:
                pkt.logprobs = pkt.logprobs[:trim]
        return pkt

    def _decode(self) -> tuple[torch.Tensor, list[list[float]], list[list[float]]]:
        """
        Decode from active lanes. The number of decoding
        iterations performed is limited by the sync
        frequency. Decoding iterations will not sync the
        cuda streams.

        Returns:
            tuple[torch.Tensor, list[list[float]], list[list[float]]]:
                The decoded tokens and their logprobs both with and
                without temperature and top_p adjustments. The
                shape of the data returned is (nactive, niter)
                where niter is the number of decoding
                iterations performed.
        """
        assert self.nactive > 0

        nactive = self.nactive
        seqlen = self.seqlen
        blktbl = self.blktbl
        tokens = self.tokens

        idx = bisect.bisect_left(self.graph_batch_sizes, nactive)
        assert idx < len(self.graphs)
        dg = self.graphs[idx]
        assert dg.state.batch_size >= nactive

        maxsl = int(seqlen.max().item())
        assert maxsl < self.max_seq
        sync_freq = self.gen_args.sync_freq
        nsteps = min(self.max_seq - maxsl, sync_freq)

        output = torch.empty(
            (nactive, nsteps),
            dtype=torch.int,
            device=self.device,
        )
        logprobs = torch.empty(
            (nactive, nsteps) if self.gen_logprobs else (0,),
            dtype=torch.float32,
            device=self.device,
        )
        sample_logprobs = torch.empty(
            (nactive, nsteps) if self.gen_logprobs else (0,),
            dtype=torch.float32,
            device=self.device,
        )

        dg.state.set_actual_batch_size(nactive)

        for i in range(nsteps):
            dg.state.copy_inputs(blktbl, tokens, seqlen)
            if dg.graph:
                dg.graph.replay()
                logits: torch.Tensor = dg.logits  # type: ignore
            else:
                logits = fwd.decode(self.model, self.tp_mesh, dg.state)
            logits = logits[:nactive]

            new_tokens, new_sample_logprobs = self._sample(
                self.temps[:nactive], self.top_ps[:nactive], logits
            )
            output[:, i] = new_tokens
            if self.gen_logprobs:
                logprobs[:, i] = -cross_entropy(
                    logits,
                    new_tokens,
                    reduction="none",
                )
                sample_logprobs[:, i] = new_sample_logprobs

            tokens = dg.state.tokens
            seqlen = dg.state.seqlen
            blktbl = dg.state.blktbl
            tokens[:nactive].copy_(new_tokens)
            seqlen[:nactive].add_(1)

        self.tokens[:nactive].copy_(tokens[:nactive])
        self.seqlen[:nactive].copy_(seqlen[:nactive])

        return output, logprobs.tolist(), sample_logprobs.tolist()

    def _prefill(
        self,
        nprefill: int,
    ) -> tuple[torch.Tensor, list[list[float]], list[list[float]], torch.Tensor]:
        """
        Fill the kv-caches for the added lanes and predict
        one new token per lane.

        Returns:
            ``(tokens, logprobs, sample_logprobs, toolong)``
            where ``toolong`` is a tensor of booleans indicating
            which of the new lanes are already at their
            length limit.
        """

        new_nactive = self.nactive + nprefill

        host_cache = self.host_cache
        gpu_cache = self.cache

        cache_lanes: list[DynCacheLane] = []
        seq_info: list[tuple[int, int]] = []
        start_pos: list[int] = []
        tokens_list: list[int] = []
        seqlen_list: list[int] = []
        maxlen_list: list[int] = []
        temps_list: list[float] = []
        top_ps_list: list[float] = []
        for i in range(self.nactive, new_nactive):
            lane = self.lane[i]
            toks = lane.all_tokens()
            blks = lane.blocks
            assert toks, "Packet.tokens must be populated with FastGen"
            assert len(toks) < self.max_seq

            cache_lane = host_cache.prepare_lane(gpu_cache, blks, toks)
            cache_lanes.append(cache_lane)
            cached = host_cache.node_len * cache_lane.ready_count
            if cached == len(toks):
                # avoid an empty sequence in the batch
                # by re-processing the last token
                cached -= 1

            seq_info.append((cached, len(toks) - cached))
            start_pos.append(len(tokens_list))
            tokens_list.extend(toks[cached:])

            self.blktbl[i, : len(blks)] = self.tensor(blks)
            seqlen_list.append(len(toks))
            maxlen_list.append(lane.maxlen)
            temps_list.append(
                lane.temperature
                if lane.temperature is not None
                else self.gen_args.temperature
            )
            top_ps_list.append(
                lane.top_p  #
                if lane.top_p is not None
                else self.gen_args.top_p,
            )
            if top_ps_list[-1] != 1:
                self.top_p_lanes.add(i)

        dyn_cache = DynCache(
            gpu_cache=gpu_cache,
            host_cache=host_cache,
            cache_lanes=cache_lanes,
        )

        # split prefill in multiple calls to avoid
        # excessive memory consumption
        fst_tok, lst_tok = 0, self.max_prefill
        fst_idx = 0
        blktbl = self.blktbl[self.nactive : new_nactive]
        logits_list: list[torch.Tensor] = []
        logprobs: list[list[float]] = []
        if self.gen_prompt_logprobs:
            logprobs = [[] for _ in start_pos]
        while lst_tok < len(tokens_list):
            lst_idx = bisect.bisect_left(start_pos, lst_tok, fst_idx)
            assert 0 < lst_idx <= nprefill
            split_idx = lst_idx - 1
            cached, ntoks = seq_info[split_idx]
            new_ntoks = lst_tok - start_pos[split_idx]
            assert 0 < new_ntoks <= ntoks
            if new_ntoks == ntoks:
                new_ntoks -= 1
                lst_tok -= 1
            seq_info[split_idx] = (cached, new_ntoks)
            tokens = self.tensor(tokens_list[fst_tok:lst_tok])
            dyn_cache.disable("page_out")
            logits, logprobs_chunk = fwd.prefill(
                model=self.model,
                coll=self.tp_coll,
                token_values=tokens,
                seq_info=seq_info[fst_idx:lst_idx],
                block_tbl=blktbl[fst_idx:lst_idx],
                block_len=self.block_len,
                cache=dyn_cache,
                cache_shard=self.cache_shard,
                logprobs=self.gen_prompt_logprobs,
            )
            if logprobs_chunk is not None:
                for i, lp in enumerate(logprobs_chunk):
                    logprobs[fst_idx + i].extend(lp)
                new_ntoks -= 1
                lst_tok -= 1
            dyn_cache.disable("page_in")
            logits_list.append(logits[:-1])
            cached += new_ntoks
            ntoks -= new_ntoks
            seq_info[split_idx] = (cached, ntoks)
            start_pos[split_idx] += new_ntoks
            fst_idx = split_idx
            fst_tok = lst_tok
            lst_tok += self.max_prefill

            if self.profile_state != "inactive":
                # to avoid recording massive prefills
                self._profile_step()

        tokens = self.tensor(tokens_list[fst_tok:])
        dyn_cache.enable("page_out")
        logits, logprobs_chunk = fwd.prefill(
            model=self.model,
            coll=self.tp_coll,
            token_values=tokens,
            seq_info=seq_info[fst_idx:],
            block_tbl=blktbl[fst_idx:],
            block_len=self.block_len,
            cache=dyn_cache,
            cache_shard=self.cache_shard,
            logprobs=self.gen_prompt_logprobs,
        )
        if logprobs_chunk is not None:
            for i, lp in enumerate(logprobs_chunk):
                logprobs[fst_idx + i].extend(lp)
        logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)
        seqlen = self.tensor(seqlen_list)
        maxlen = self.tensor(maxlen_list)
        temps = torch.tensor(
            temps_list,
            dtype=torch.float,
            device=self.device,
        )
        top_ps = torch.tensor(
            top_ps_list,
            dtype=torch.float,
            device=self.device,
        )

        host_cache.tick()
        assert self.cache_ready is None
        self.cache_ready = dyn_cache.host_cache_ready()

        next_tokens, sample_logprobs_ = self._sample(temps, top_ps, logits)
        if self.gen_logprobs:
            token_logprobs = (
                -cross_entropy(
                    logits,
                    next_tokens,
                    reduction="none",
                )
            )[:, None].tolist()
            if self.gen_prompt_logprobs:
                for i, lp in enumerate(token_logprobs):
                    logprobs[i].extend(lp)
            else:
                logprobs = token_logprobs
            sample_logprobs = sample_logprobs_[:, None].tolist()
        else:
            sample_logprobs = []

        self.tokens[self.nactive : new_nactive] = next_tokens
        self.seqlen[self.nactive : new_nactive] = seqlen + 1
        self.maxlen[self.nactive : new_nactive] = maxlen
        self.temps[self.nactive : new_nactive] = temps
        self.top_ps[self.nactive : new_nactive] = top_ps
        self.nactive = new_nactive

        toolong = (seqlen + 1) >= maxlen
        return next_tokens[:, None], logprobs, sample_logprobs, toolong

    def _sample(
        self,
        temps: torch.Tensor,
        top_ps: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gen_args.use_sampling:
            probs = torch.softmax(logits / temps[:, None], dim=-1)
            if not self.top_p_lanes:
                next_tokens, next_tokens_prob = sampling.sample(probs)
            else:
                next_tokens, next_tokens_prob = sampling.top_p(probs, top_ps[:, None])
            # torch.log is numerically fine because probs
            # are in float32 and it handles up to 1e-45
            next_tokens_logprob = torch.log(next_tokens_prob)
        else:
            next_tokens = torch.argmax(logits, dim=-1)
            next_tokens_logprob = torch.zeros_like(next_tokens, dtype=logits.dtype)
        return next_tokens, next_tokens_logprob

    def _kill_lane(self, lane: int, park: bool = False) -> None:
        """
        Kill a previously active generation lane. In case the
        lane killed is not the last one, the last one takes
        its place. If the ``park`` argument is True, the victim
        lane is parked.
        """
        last = self.nactive - 1
        self.nactive = last

        self.lane[lane].free(self.free_blocks)

        if park:
            self.parking.append(self.lane[lane])

        if lane == last:
            self.lane[lane] = Lane("dead")
            self.seqlen[lane] = 0
            self.maxlen[lane] = 0
            self.temps[lane] = 0
            self.top_ps[lane] = 0
            self.blktbl[lane] = 0
            self.top_p_lanes.discard(lane)
            return

        self.lane[lane] = self.lane[last]
        self.lane[last] = Lane("dead")

        self.tokens[lane] = self.tokens[last]
        self.seqlen[lane] = self.seqlen[last]
        self.seqlen[last] = 0
        self.maxlen[lane] = self.maxlen[last]
        self.maxlen[last] = 0
        self.temps[lane] = self.temps[last]
        self.temps[last] = 0
        self.top_ps[lane] = self.top_ps[last]
        self.top_ps[last] = 0
        self.blktbl[lane] = self.blktbl[last]
        self.blktbl[last] = 0
        if last in self.top_p_lanes:
            self.top_p_lanes.remove(last)
            self.top_p_lanes.add(lane)

    def tensor(self, seq: Iterable[int]) -> torch.Tensor:
        return torch.tensor(seq, dtype=torch.int, device=self.device)

    def _log_cuda_mem(self) -> None:
        used_gb = torch.cuda.memory_allocated() / 1e9
        if self.last_used_gb is not None:
            delta_gb = used_gb - self.last_used_gb
            logger.info(f"GPU memory: {used_gb:.3f}GB ({delta_gb:+.3f}GB)")
        else:
            logger.info(f"GPU memory: {used_gb:.3f}GB")
        self.last_used_gb = used_gb
