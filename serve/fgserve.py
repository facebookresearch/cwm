# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from uuid import uuid4

import torch
import torch.distributed
from flask import Flask, Response, jsonify, request
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from cwm.common.environment import (
    get_world_size,
    init_torch_distributed,
    set_seed,
    setup_env,
)
from cwm.common.params import load_from_cli
from cwm.fastgen.generate import FastGen, GenArgs
from cwm.fastgen.utils.loading import build_fastgen_model, build_tokenizer_from_ckpt
from cwm.logging.logger import initialize_logger, set_root_log_level
from cwm.rl.lib.impgen import ImpGen
from cwm.text.datatypes import CWMChatMessage, MessageBase
from cwm.text.tokenizers import InstructTokenizer, build_tokenizer
from serve.openai_api import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionsRequest,
    CompletionUsage,
)
from serve.tools import (
    Tool,
    format_calls,
    parse_calls,
    prepare_tools,
    tool_from_call_id,
)

logger = logging.getLogger()


@dataclass
class TokenizerArgs:
    name: str = ""
    path: str | None = None


@dataclass
class FastGenArgs(GenArgs):
    # Add to base class
    vocab_parallel: bool = False
    loss_parallel: bool = False

    # Overriding defaults from base class
    use_sampling: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    max_batch: int = 256
    sync_freq: int = 1


@dataclass
class FGServeArgs:
    dump_dir: str = ""
    log_level: str = "info"
    request_log: str | None = None
    port: int = 5678

    perf_log_freq: int = 60

    checkpoint_dir: str = ""
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)

    # default values which can be overwritten per request
    gen_args: FastGenArgs = field(default_factory=FastGenArgs)

    seed: int = 17


def convert_to_cwm_message(msg: ChatMessage, tools: list[Tool]) -> MessageBase:
    body = msg.content_str
    if msg.tool_calls:
        if body and body[-1] != "\n":
            body += "\n"
        body += format_calls(msg.tool_calls, tools)

    return CWMChatMessage.from_dict(
        {
            "source": msg.role,
            "body": body,
            "eot": True,
            "tool": tool_from_call_id(msg.tool_call_id or ""),
        }
    )


def setup_mesh() -> tuple[DeviceMesh, torch.distributed.ProcessGroup]:
    tp_size = get_world_size()

    tp_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(tp_size,),
        mesh_dim_names=("tp",),
    )

    # FastGen creates cuda graphs which requires NCCL process groups,
    # but ImpGen requires moodist because it's using moodist queues,
    # so we create them here
    md_group = torch.distributed.new_group(range(tp_size), backend="moodist")

    return tp_mesh, md_group


def start_flask_server(
    tokenizer: InstructTokenizer,
    port: int,
    g: ImpGen,
    request_log: str | None,
    fingerprint: str | None,
) -> None:
    app = Flask(__name__)

    @app.route("/chat/completions", methods=["POST"])
    def handle_completions():
        request_json = request.get_json()
        rq = ChatCompletionRequest(**request_json)
        reasoning = (
            rq.reasoning is not None
            and rq.reasoning.enabled
            or rq.reasoning_effort is not None
        )

        tools_msg, tools = prepare_tools(rq.tools or [])
        if tools_msg:
            last_system_msg = -1
            for i, m in enumerate(rq.messages):
                if m.role == "system":
                    last_system_msg = i
            rq.messages.insert(last_system_msg + 1, tools_msg)
        msgs = [convert_to_cwm_message(m, tools) for m in rq.messages]
        prompt_tokens = tokenizer.encode_prompt_dialog(msgs, think=reasoning)

        completions: list[list[int]] = []
        for _ in range(rq.n):
            pkt = g.generate(
                tokens=prompt_tokens,
                max_gen=rq.max_tokens or 8192,
                temperature=rq.temperature or None,
                top_p=rq.top_p,
                stop_str=rq.stop,
            )
            completions.append(pkt.tokens)
        rsp = ChatCompletionResponse(
            system_fingerprint=fingerprint,
            model=rq.model or "",
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=len(prompt_tokens),
                total_tokens=len(prompt_tokens),
            ),
        )
        assert rsp.usage is not None  # for typing
        outputs: list[str] = []
        for ix, tokens in enumerate(completions):
            rsp.usage.completion_tokens += len(tokens)
            rsp.usage.total_tokens += len(tokens)
            if tokens[-1] in tokenizer.stop_tokens:
                tokens = tokens[:-1]
            reasoning_content: str | None = None
            content = tokenizer.decode(tokens)
            outputs.append(content)
            if reasoning:
                chunks = content.split("\n</think>\n", 1)
                if len(chunks) == 2:
                    reasoning_content, content = chunks
            content, tool_calls = parse_calls(content, tools)
            rsp.choices.append(
                ChatCompletionChoice(
                    index=ix,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls,
                    ),
                )
            )

        if request_log is not None:
            with open(request_log, "a") as f:
                call = {
                    "request": request_json,
                    "response": rsp.model_dump(),
                    "model_prompt": tokenizer.decode(
                        prompt_tokens,
                        cut_at_stop_tokens=False,
                    ),
                    "model_output": outputs,
                }
                f.write(json.dumps(call) + "\n")
                f.flush()

        if not rq.stream:
            return Response(
                response=rsp.model_dump_json(),
                mimetype="application/json",
            )
        else:
            # fake stream
            srsp = ChatCompletionStreamResponse.from_chat(rsp)
            return Response(
                response="".join(
                    [
                        f"data: {srsp.model_dump_json()}\n\n",
                        "data: [DONE]\n\n",
                    ]
                ),
                mimetype="text/event-stream",
            )

    @app.route("/completions", methods=["POST"])
    def handle_text_completions():
        rq = CompletionsRequest(**request.get_json())
        rid = uuid4().hex

        prompt_tokens = rq.prompt

        is_tokenized = isinstance(prompt_tokens, list)

        if not is_tokenized:
            prompt_tokens = tokenizer.encode(prompt_tokens, bos=True)

        completions: list[list[int]] = []
        for _ in range(rq.n):
            pkt = g.generate(
                tokens=prompt_tokens,
                max_gen=rq.max_tokens,
                temperature=rq.temperature,
                top_p=rq.top_p,
                stop_str=rq.stop,
            )
            completions.append(pkt.tokens)
        rsp = {
            "id": rid,
            "choices": [],
            "created": int(time.time()),
            "model": rq.model,
            "object": "text_completion",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": len(prompt_tokens),
                "total_tokens": len(prompt_tokens),
            },
            "system_fingerprint": fingerprint,
        }
        for ix, tokens in enumerate(completions):
            # TODO use the logprobs in the response
            rsp["usage"]["completion_tokens"] += len(tokens)
            rsp["usage"]["total_tokens"] += len(tokens)
            rsp["choices"].append(
                {
                    "index": ix,
                    "text": tokenizer.decode(tokens),
                    "tokens": tokens,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            )
        return jsonify(rsp)

    app.run(host="127.0.0.1", port=port, threaded=True)


def serve(args: FGServeArgs) -> None:
    init_torch_distributed(backend="nccl", timeout=600)
    set_seed(args.seed)

    tp_mesh, md_group = setup_mesh()
    rank = tp_mesh.get_rank()

    if args.tokenizer.name:
        tokenizer = build_tokenizer(args.tokenizer.name, args.tokenizer.path)
    else:
        tokenizer = build_tokenizer_from_ckpt(args.checkpoint_dir)

    assert hasattr(tokenizer, "encode_message")

    model = build_fastgen_model(
        world_mesh=tp_mesh,
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
        tp_mesh=tp_mesh,
    )
    g = ImpGen(fg, md_group.rank(), md_group)

    if rank == 0:
        flask_server_thread = threading.Thread(
            target=start_flask_server,
            kwargs=dict(
                tokenizer=tokenizer,
                port=args.port,
                g=g,
                request_log=args.request_log,
                fingerprint=args.checkpoint_dir,
            ),
        )
        flask_server_thread.start()

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


if __name__ == "__main__":
    initialize_logger()
    args = load_from_cli(FGServeArgs, from_config_file=True, with_preset=True)
    set_root_log_level(args.log_level)
    setup_env(mp_spawn_method="forkserver")
    serve(args)
