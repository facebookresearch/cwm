# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self, cast

import requests

from cwm.text.tokenizers import CWMInstructTokenizer, build_tokenizer

_START_MARKER = "  # << START_OF_TRACE"


class CWMTraceEvent(Enum):
    CALL = auto()
    RETURN = auto()
    LINE = auto()
    EXCEPTION = auto()


@dataclass
class CWMTraceFrame:
    event: CWMTraceEvent
    source_line: str
    # local variables in diff-based representation
    local_vars: dict[str, str]
    arg: str | None

    line_no: int | None
    final: bool = False

    prev: Self | None = None
    tokens_: list[int] | None = None


class CWMDebugger:
    def __init__(
        self,
        source: str,
        tokenizer: CWMInstructTokenizer,
        host: str,
        port: int,
        temperature: float = 0.2,
    ):
        self.source = source
        self.tokenizer = tokenizer
        self.host = host
        self.port = port
        self.temperature = temperature

        # Strip start marker (usually not predicted by the model) from lines
        # used for lookup
        self.source_lines = [line.rstrip(_START_MARKER) for line in source.splitlines()]
        self.frames: list[CWMTraceFrame] = []

    def reset(self) -> CWMTraceFrame:
        self.frames.clear()
        self._step(CWMTraceEvent.CALL)
        return self.current_frame

    @property
    def done(self) -> bool:
        return self.current_frame.final

    @property
    def current_frame(self) -> CWMTraceFrame:
        return self.frames[-1]

    @property
    def local_vars(self) -> dict[str, str]:
        """
        Return local variables of the current frame.

        In contrast to `current_frame.local_vars`, we'll try to resolve values
        abbreviated due to the diff-based representation.
        """

        f = self.current_frame
        if f.event in {CWMTraceEvent.RETURN, CWMTraceEvent.EXCEPTION}:
            # No local vars predicted for these events, use previous ones
            assert f.prev is not None
            f = f.prev

        # Unroll diff-based locals representation and replace placeholder '..'
        # with correct values by iterating through frames in reverse order
        lv = f.local_vars.copy()
        for k, v in lv.items():
            if v != "..":
                continue
            # Same as last occurrence
            frame = f.prev
            while frame is not None:
                if (v := frame.local_vars.get(k, "..")) != "..":
                    lv[k] = v
                    break
                if frame.event in {CWMTraceEvent.CALL, CWMTraceEvent.RETURN}:
                    # Stop iteration on context change
                    break
                frame = frame.prev
        return lv

    def next(self) -> CWMTraceFrame:
        if self.done:
            return self.current_frame

        # Next event; if it's a call event, step over it
        self._step()
        if self.current_frame.event == CWMTraceEvent.CALL:
            self._step(CWMTraceEvent.RETURN)
            self._step()
        return self.current_frame

    def step(self) -> CWMTraceFrame:
        if self.done:
            return self.current_frame

        # If last event is a call event, ensure we step into the function
        # rather than stepping over it
        if self.current_frame.event == CWMTraceEvent.CALL:
            self._step(CWMTraceEvent.LINE)
        else:
            self._step()
        return self.current_frame

    def step_out(self) -> CWMTraceFrame:
        # Continue until return event, then step to next line
        while not self.done:
            if self.current_frame.event == CWMTraceEvent.CALL:
                # Do not recurse into functions
                self._step(CWMTraceEvent.RETURN)
            else:
                self._step()

            if self.current_frame.event == CWMTraceEvent.RETURN:
                self._step()
                break
        return self.current_frame

    def step_back(self) -> CWMTraceFrame:
        if len(self.frames) > 1:
            self.frames.pop()
        if len(self.frames) == 0:
            return self.reset()
        return self.current_frame

    def _event_token(self, event: CWMTraceEvent) -> int:
        match event:
            case CWMTraceEvent.CALL:
                return self.tokenizer.call_sep_id
            case CWMTraceEvent.RETURN:
                return self.tokenizer.return_sep_id
            case CWMTraceEvent.LINE:
                return self.tokenizer.line_sep_id
            case CWMTraceEvent.EXCEPTION:
                return self.tokenizer.exception_sep_id

    def _step(self, event: CWMTraceEvent | None = None) -> None:
        # Build prompt: context + existing frames
        tokens = [self.tokenizer.bos_id, self.tokenizer.trace_context_start_id]
        tokens += self.tokenizer.encode(self.source)
        tokens += [self.tokenizer.frame_sep_id]

        for frame in self.frames:
            if frame.tokens_ is None:
                frame.tokens_ = []
                frame.tokens_ += [self._event_token(frame.event)]
                if frame.event in {CWMTraceEvent.CALL, CWMTraceEvent.LINE}:
                    frame.tokens_ += self.tokenizer.encode(json.dumps(frame.local_vars))
                frame.tokens_ += [self.tokenizer.action_sep_id]
                frame.tokens_ += self.tokenizer.encode(frame.source_line)
                if frame.event in {CWMTraceEvent.RETURN, CWMTraceEvent.EXCEPTION}:
                    frame.tokens_ += [self.tokenizer.arg_sep_id]
                    frame.tokens_ += self.tokenizer.encode(json.dumps(frame.arg))

            tokens += frame.tokens_
            tokens += [self.tokenizer.frame_sep_id]

        if event is not None:
            tokens += [self._event_token(event)]

        rep = requests.post(
            url=f"http://{self.host}:{self.port}/completions",
            json={
                "prompt": tokens,
                "max_tokens": 2048,
                "temperature": self.temperature,
                "top_p": 0.95,
                "stop": self.tokenizer.FRAME_SEP_ID,
            },
        )

        # Parse response:
        # Event
        rep_tokens = deque(rep.json()["choices"][0]["tokens"])
        if event is None:
            for evt in CWMTraceEvent:
                if rep_tokens[0] == self._event_token(evt):
                    event = evt
                    break
            else:
                if rep_tokens[0] == self.tokenizer.eos_id:
                    assert len(self.frames) > 0
                    self.current_frame.final = True
                    return
                prompt_str = self.tokenizer.decode(tokens, cut_at_stop_tokens=False)
                rep_str = self.tokenizer.decode(
                    list(rep_tokens), cut_at_stop_tokens=False
                )
                raise RuntimeError(
                    f"Unexpected reply: {prompt_str=}, {rep_tokens=}, {rep_str=}"
                )

            rep_tokens.popleft()

        # Locals
        local_vars_tokens: list[int] = []
        while (tok := rep_tokens.popleft()) != self.tokenizer.action_sep_id:
            local_vars_tokens.append(tok)

        # Source line
        source_line_tokens: list[int] = []
        while (tok := rep_tokens.popleft()) not in {
            self.tokenizer.frame_sep_id,
            self.tokenizer.arg_sep_id,
        }:
            source_line_tokens.append(tok)

        # Argument for return and exception frames
        arg_tokens: list[int] = []
        if tok == self.tokenizer.arg_sep_id:
            while (tok := rep_tokens.popleft()) not in {self.tokenizer.frame_sep_id}:
                arg_tokens.append(tok)

        if local_vars_tokens:
            local_vars_str = self.tokenizer.decode(local_vars_tokens)
            try:
                local_vars = json.loads(local_vars_str)
            except json.decoder.JSONDecodeError:
                local_vars = {"_DECODING_ERROR_": local_vars_str}
        else:
            local_vars = {}

        if arg_tokens:
            arg_str = self.tokenizer.decode(arg_tokens)
            try:
                arg = json.loads(arg_str)
            except json.decoder.JSONDecodeError:
                arg = arg_str
        else:
            arg = None

        source_line = self.tokenizer.decode(source_line_tokens)
        # Attempt to find matching source line
        line_no: int | None = None
        try:
            line_no = (
                self.source_lines.index(source_line.rstrip("\n").rstrip(_START_MARKER))
                + 1
            )
        except ValueError:
            pass

        self.frames.append(
            CWMTraceFrame(
                event=event,
                source_line=source_line,
                local_vars=local_vars,
                arg=arg,
                line_no=line_no,
                final=False,
                prev=self.frames[-1] if self.frames else None,
            )
        )


def main(
    host: str, port: int, tokenizer_path: str, temperature: float, context: str
) -> None:
    tokenizer = cast(
        CWMInstructTokenizer,
        build_tokenizer(
            name="cwm_instruct",
            path=tokenizer_path,
        ),
    )

    with open(context) as f:
        src = f.read()

    dbg = CWMDebugger(src, tokenizer, host, port, temperature)
    dbg.reset()
    print_ = True

    print("===== CONTEXT =====\n")
    print(src)
    print("\n===== SESSION START =====\n")

    while not dbg.done:
        if print_:
            frame = dbg.current_frame
            print("->", frame.source_line.rstrip())
            if frame.event in (CWMTraceEvent.RETURN, CWMTraceEvent.EXCEPTION):
                print(frame.arg)
            print_ = False

        try:
            action = input(">> ")
        except EOFError:
            break

        if action in ("h", "help", "?"):
            print("Actions: ")
            print("s, step      step next/into")
            print("n, next      next line")
            print("o, out       step out")
            print("b, back      step back")
            print("p, print     print local variable (or all)")
            print("reset        reset, restart session")
            print("q, quit      quit")
        elif action in ("s", "step"):
            dbg.step()
            print_ = True
        elif action in ("n", "next"):
            dbg.next()
            print_ = True
        elif action in ("o", "out"):
            dbg.step_out()
            print_ = True
        elif action in ("b", "back"):
            dbg.step_back()
            print_ = True
        elif action in ("reset",):
            dbg.reset()
            print_ = True
        elif (
            action in ("p", "print")
            or action.startswith("p ")
            or action.startswith("print ")
        ):
            lv = dbg.local_vars
            if action.startswith("p "):
                var = action.removeprefix("p ")
            elif action.startswith("print "):
                var = action.removeprefix("print ")
            else:
                var = ""

            if not var:
                print(lv)
                continue
            elif var in lv:
                print(lv[var])
            else:
                print(f"No such variable: {var}")
        elif action in ("q", "quit"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host of fgserve instance"
    )
    parser.add_argument(
        "--port", type=int, default=5678, help="Port of fgserve instance"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument("tokenizer", type=str, help="Path to CWM tokenizer file")
    parser.add_argument("context", type=str, help="Input file")
    args = parser.parse_args()

    main(args.host, args.port, args.tokenizer, args.temperature, args.context)
