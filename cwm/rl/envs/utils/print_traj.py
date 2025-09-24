# Copyright (c) Meta Platforms, Inc. and affiliates.

import rich
from rich.pretty import pprint
from rich.table import Table

from cwm.rl.envs.api import Trajectory
from cwm.rl.lib.dump_traj import trajectory_from_dict
from cwm.text.tokenizers import Tokenizer


def print_traj(traj: Trajectory, max_length: int = 10) -> None:
    """
    Recommended way of inspecting trajectories, using rich pretty printing.
    By setting max_length we truncate long lists of tokens.
    """
    pprint(vars(traj), max_length=max_length)


def print_traj_str(traj: Trajectory, tokenizer: Tokenizer) -> None:
    """
    Print the trajectory as strings (one per context).
    Using rich.print we get nice syntax highlighting for
    code enclosed in triple backticks.
    """
    for ctx_toks in traj.tokens:
        rich.print(tokenizer.decode(ctx_toks, cut_at_stop_tokens=False))


def zip_traj(
    traj: Trajectory, tokenizer: Tokenizer | None = None
) -> list[tuple[tuple[str, int, str, float]]] | list[tuple[tuple[int, str, float]]]:
    """
    Zip the trajectory into lists containing for each token a tuple with the
    token as a string (if tokenizer is not None), the token id, the token source,
    and token reward.
    """

    if tokenizer is not None:
        # Decode each token separately
        decoded = [[repr(tokenizer.decode([t])) for t in toks] for toks in traj.tokens]
        tozipd = (decoded, traj.tokens, traj.source, traj.rewards)
        return [
            tuple(zip(*lists, strict=False)) for lists in zip(*tozipd, strict=False)
        ]  # type: ignore
    else:
        tozip = (traj.tokens, traj.source, traj.rewards)
        return [tuple(zip(*lists, strict=False)) for lists in zip(*tozip, strict=False)]  # type: ignore


def print_traj_table(traj: Trajectory, tokenizer: Tokenizer | None = None) -> None:
    """
    Print the trajectory as a table.
    The columns are token str (if tokenizer is not None), token id, source, and reward.
    There is one row per token in the trajectory, and a horizontal line for each context switch.
    """

    z = zip_traj(traj, tokenizer)  # type: ignore

    table = Table(title="Trajectory")
    if tokenizer is not None:
        table.add_column("Token str", justify="left", style="cyan", no_wrap=True)
    table.add_column("Token id", justify="left", style="magenta")
    table.add_column("Source", justify="center", style="yellow")
    table.add_column("Reward", justify="left", style="green")

    for ctx in z:
        table.add_section()

        for row in ctx:
            table.add_row(*(str(x) for x in row))

    rich.print(table)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    from rich.console import Console
    from rich.text import Text

    parser = argparse.ArgumentParser("Pretty-print trajectories")
    parser.add_argument(
        "--text", action="store_true", help="Print observation/action strings only"
    )
    parser.add_argument(
        "file", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    args = parser.parse_args()

    d = json.load(args.file)
    assert "transitions" in d, "Not a trajectory?"

    if not args.text:
        # This better be a full trajectory
        traj = trajectory_from_dict(d)
        print_traj(traj)
        sys.exit(0)

    # Get observation/action sequence
    has_chat_messages: bool = False
    text = Text()

    for tr in d["transitions"]:
        if "context_switch" in tr:
            assert not tr["context_switch"], "Context switch not supported"
            # TODO? print something about this?

        if act := tr.get("action_str"):
            if has_chat_messages:
                if not act.endswith("<|eot_id|>"):
                    act = act + "<|eot_id|>"  # XXX

                act = act.replace("<|start_header_id|>", "\n----------\n")
                act = act.replace("<|end_header_id|>", "\n----------\n")
                act = act.replace("|>", "|>\n")

                text.append(act, style="bold")
            else:
                pprint(act)

        if obs := tr.get("observation_str"):
            if "<|start_header_id|>" in obs:
                has_chat_messages = True
            if has_chat_messages:
                if obs.startswith("<|eot_id|>"):
                    obs = obs[len("<|eot_id|>") :]
                obs = obs.replace("<|start_header_id|>", "\n----------\n")
                obs = obs.replace("<|end_header_id|>", "\n----------\n")
                obs = obs.replace("|>", "|>\n")

                text.append(obs)
            else:
                pprint(obs)

    if not has_chat_messages:
        sys.exit(0)

    Console().print(text)
