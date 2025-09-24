# Copyright (c) Meta Platforms, Inc. and affiliates.


import json
import shlex
import textwrap

from .default_configs import DEFAULT_PYTHON_PATH


def restore_env(d: dict, bash_env_only: bool = False) -> str:
    """
    Build a bash script that may be run to restore the
    bash environment for the argument task dict.
    """
    if "entrypoint_script" in d:
        # Some images (e.g. Activ) use this
        return d["entrypoint_script"]
    elif "init_commands" in d and not bash_env_only:
        return "\n".join(c["cmd"] for c in d["init_commands"])
    elif "snapshot" in d and d["snapshot"]:
        try:
            snapshot = json.loads(d["snapshot"])
        except json.JSONDecodeError:
            return ""
        exports = [
            f"export {var}={shlex.quote(val)}"
            for (var, val) in json.loads(snapshot["env_json"]).items()
        ]
        bash_env = snapshot["bash_env"]
        exports_str = "\n".join(exports)
        if bash_env_only:
            return exports_str
        snapshot_command = f"{bash_env}\n{exports_str}"
        return snapshot_command
    else:
        return ""


def get_server_python_path(d: dict) -> str:
    return DEFAULT_PYTHON_PATH


def retain_only_current_branch_ancestor_commits(repo_root_path: str) -> str:
    """Remove git logs to prevent leakage from future commits."""
    git = f"git -C {repo_root_path}"

    return "\n".join(
        [
            # remove all remotes
            f"{git} remote | xargs -n1 {git} remote remove",
            # delete all branches except the current one
            textwrap.dedent(
                f"""\
            cur=$({git} symbolic-ref --quiet --short HEAD);
            {git} for-each-ref --format='%(refname)' refs \\
             | awk -v cur="$cur" '$0 != "refs/heads/" cur' \\
             | xargs -n1 -I{{}} {git} update-ref -d {{}}
        """
            ),
            # unset upstream for current branch (ignore errors if no upstream exists)
            f"{git} branch --unset-upstream",
            # purge reflog
            f"{git} reflog expire --expire=now --all",
            f"{git} gc --prune=now",
        ]
    )
