# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Light-weight evaluation on SWE-Bench style tasks"""

import base64
import json
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import modal

from ..modal_backend import MODAL_RETRY, default_secret, global_modal_app
from .utils.constants import (
    APPLY_PATCH_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    PASS_TO_PASS,
)
from .utils.grading import get_eval_report
from .utils.test_spec import TestSpec


@dataclass(frozen=True)
class EvalResult:
    outcome: Literal["pass", "fail", "env_error", "timeout"]
    message: str
    duration: float | None = None
    returncode: int | None = None


# Some additional porting logic from official SWE-bench

# Lets keep trying to apply the patch! Log the outcomes.
GIT_APPLY_CMDS = [
    # Originals
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
    "git apply --allow-empty -v",
    # Whitespace handling
    "git apply --ignore-whitespace --verbose",
    "git apply --whitespace=fix",
    # Path variations
    "git apply -p0 --verbose",
    "git apply -p2 --verbose",
    # Force application
    "git apply --verbose --force",
    "patch --batch --force -p1 -i",
    # Higher fuzz
    "patch --batch --fuzz=10 -p1 -i",
    # Combined approaches
    "git apply --ignore-whitespace --verbose --force",
    "patch --batch --force --forward -p1 -i",
    # Last resort
    "patch --batch --force --forward --no-backup-if-mismatch --reject-file=/dev/null -p1 -i",
]
DOCKER_PATCH = "/tmp/patch.diff"
LOG_REPORT = "report.json"


# Let's make this consistent with the container backend's name
MODAL_APPNAME = "swe-remote"


def eval_instance_default(
    spec: TestSpec,
    pred_patch: str,
    eval_dir: str | Path,
    timeout: int | float,
    workdir: str = "/testbed",
    # Soft / hard limit
    cpu: tuple[float, float] | None = (0.125, 4.0),
    memory: tuple[int, int] | None = (1024, 16384),
    rm_dir_after_eval: bool = False,
    backend: str = "modal",
) -> EvalResult:
    """
    A simplified version of run_evaluation in SWE-bench.
    """

    if backend == "modal":
        return eval_instance_modal(
            spec=spec,
            pred_patch=pred_patch,
            eval_dir=eval_dir,
            timeout=timeout,
            workdir=workdir,
            cpu=cpu,
            memory=memory,
            rm_dir_after_eval=rm_dir_after_eval,
        )
    else:
        raise AssertionError(
            f"Unsupported backend: {backend}. Supported backends are 'modal'"
        )


def eval_instance_modal(
    spec: TestSpec,
    pred_patch: str,
    eval_dir: str | Path,
    timeout: int | float,
    workdir: str = "/testbed",
    cpu: tuple[float, float] | None = (0.125, 4.0),
    memory: tuple[int, int] | None = (1024, 16384),
    rm_dir_after_eval: bool = False,
) -> EvalResult:
    if isinstance(eval_dir, str):
        eval_dir = Path(eval_dir)
    timeout = int(timeout)
    sandbox: modal.sandbox.Sandbox | None = None
    try:
        # Copy model prediction as patch file to container
        eval_dir.mkdir(parents=True, exist_ok=True)
        patch_file = eval_dir / "patch.diff"
        patch_file.write_text(pred_patch)

        # Core evaluation
        # NOTE: some hacks
        # https://github.com/SWE-bench/SWE-bench/blob/006a760a95c9cc11e987884d7e311d74a16db88a/swebench/harness/modal_eval/run_evaluation_modal.py#L299-L311
        eval_file = eval_dir / "eval.sh"
        eval_script = spec["eval_script"].replace(
            "locale-gen", "locale-gen en_US.UTF-8"
        )
        eval_file.write_text(eval_script)

        secret = default_secret(spec["docker_url"])
        # Encode to avoid shell escaping issues
        patch_content = patch_file.read_text()
        eval_content = eval_file.read_text()
        encoded_patch = base64.b64encode(patch_content.encode("utf-8")).decode("ascii")
        encoded_eval = base64.b64encode(eval_content.encode("utf-8")).decode("ascii")

        # Avoid file mounting, copy files in using environment variables
        image = (
            modal.Image.from_registry(spec["docker_url"], secret=secret)
            .entrypoint([])
            .env(
                {
                    "PATCH_B64": encoded_patch,
                    "EVAL_SCRIPT_B64": encoded_eval,
                }
            )
            .run_commands(
                [
                    # Create patch and eval script files from base64 encoded content
                    f"echo $PATCH_B64 | base64 -d > {DOCKER_PATCH}",
                    "echo $EVAL_SCRIPT_B64 | base64 -d > /eval.sh",
                    "chmod +x /eval.sh",
                ]
            )
        )
        app = global_modal_app()

        # Activ images use entrypoint_script
        entrypoint_script = spec.get("entrypoint_script")
        if isinstance(entrypoint_script, str) and entrypoint_script:
            encoded_entrypoint = base64.b64encode(
                entrypoint_script.encode("utf-8")
            ).decode("ascii")

            image_with_entrypoint = image.env(
                {"ENTRYPOINT_SCRIPT_B64": encoded_entrypoint}
            ).run_commands(
                [
                    "echo $ENTRYPOINT_SCRIPT_B64 | base64 -d > /entrypoint.sh",
                    "chmod +x /entrypoint.sh",
                ]
            )

            @MODAL_RETRY
            def create_sandbox_with_entrypoint():
                return modal.Sandbox.create(
                    "sleep",
                    "infinity",
                    app=app,
                    image=image_with_entrypoint,
                    timeout=timeout,
                    workdir=workdir,
                    cpu=cpu,
                    memory=memory,
                )

            sandbox = create_sandbox_with_entrypoint()
            sandbox.exec("/bin/bash", "/entrypoint.sh").wait()

        else:

            @MODAL_RETRY
            def create_sandbox():
                return modal.Sandbox.create(
                    "sleep",
                    "infinity",  # keep container running
                    app=app,
                    image=image,
                    timeout=timeout,
                    workdir=workdir,
                    cpu=cpu,
                    memory=memory,
                )

            sandbox = create_sandbox()

        for cmd in spec.get("init_commands", []):
            if cmd["shell"]:
                continue  # those must run in the eval script
            proc = sandbox.exec("bash", "-c", cmd["cmd"])
            proc.wait()
            if proc.returncode != 0:
                error = proc.stderr.read()
                return EvalResult(
                    outcome="env_error",
                    message=f"Env error: Init command {cmd!r} failed: {error}",
                )

        # Apply patch for modal backend
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            git_apply_commands = git_apply_cmd.split()
            proc = sandbox.exec(*git_apply_commands, DOCKER_PATCH)
            proc.wait()
            output = proc.stdout.read()
            error = proc.stderr.read()
            if proc.returncode == 0:
                applied_patch = True
                break
        if not applied_patch:
            return EvalResult(
                outcome="fail",
                message=f"{APPLY_PATCH_FAIL}:\nstdout:\n{output}\n\nstderr:{error}",
            )

        # Run eval script, write output to logs
        run_command = f"cd {workdir}"
        # pylint hack
        if "pylint" in spec["instance_id"]:
            run_command = "PYTHONPATH="
        # increase recursion limit for testing
        run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
        # run eval script
        run_command += " && /bin/bash /eval.sh"
        run_command = f"({run_command}) 2>&1"

        eval_proc = sandbox.exec("bash", "-c", run_command)
        eval_proc.wait()
        test_output = eval_proc.stdout.read()
        timed_out = eval_proc.poll() == -1

        test_output_path = eval_dir / "test_output.txt"
        if timed_out:
            test_output = f"{test_output}\n\nTimeout error: {timeout} seconds exceeded."
        test_output_path.write_text(test_output)

        if timed_out:
            return EvalResult("timeout", message=test_output)

        pred = {
            KEY_INSTANCE_ID: spec[KEY_INSTANCE_ID].lower(),  # type: ignore
            KEY_PREDICTION: test_output,
        }

        def _from_json_or_obj(obj: list[str] | str) -> list[str]:
            """If key points to string, load with json"""
            if isinstance(obj, str):
                return json.loads(obj)
            return obj

        # Get evaluation logs

        test_parser: str | None = spec.get("test_parser")
        test_spec = {
            KEY_INSTANCE_ID: spec[KEY_INSTANCE_ID].lower(),  # type: ignore
            PASS_TO_PASS: _from_json_or_obj(spec[PASS_TO_PASS]),  # type: ignore
            FAIL_TO_PASS: _from_json_or_obj(spec[FAIL_TO_PASS]),  # type: ignore
            "test_parser": test_parser,
        }
        report: dict = get_eval_report(
            test_spec=test_spec,  # type: ignore
            prediction=pred,
            log_path=test_output_path.absolute().as_posix(),
            include_tests_status=True,
        )

        # Write report to report.json
        report_path = eval_dir / LOG_REPORT
        report_path.write_text(json.dumps(report, indent=4))
        passes = report.get(spec[KEY_INSTANCE_ID].lower(), {}).get("resolved", False)  # type: ignore
        outcome = "pass" if passes else "fail"
        try:
            message = json.dumps(report, indent=2)
        except TypeError:
            message = str(report)
        return EvalResult(outcome=outcome, message=message)  # type: ignore
    except Exception as e:
        trace = traceback.format_exc()  # no limit

        # repr() more/diff info than str()?
        error_repr = repr(e)
        error_str = str(e)

        if error_repr != error_str:
            error_msg = f"{error_str}\nRepr: {error_repr}"
        else:
            error_msg = error_str

        return EvalResult(
            outcome="env_error",
            message=f"Env error: {type(e).__name__}: {error_msg}\n\nFull traceback:\n{trace}",
        )
    finally:
        if rm_dir_after_eval:
            # Remove "eval_dir" from host
            shutil.rmtree(eval_dir.as_posix(), ignore_errors=True)
        if sandbox is not None:
            sandbox.terminate()


def eval_instance_general_modal(
    image_url: str,
    test_script: str,
    timeout: int | float,
    # some initialization script like env preparation
    setup_script: str | None = None,
    code_patch: str | None = None,
    test_patch: str | None = None,
    workdir: str = "/testbed",
    # Soft / hard limit
    cpu: tuple[float, float] | None = (0.125, 4.0),
    memory: tuple[int, int] | None = (1024, 16384),
    code_patch_path: str = "/code_patch.diff",
    test_patch_path: str = "/test_patch.diff",
    test_script_path: str = "/test_script.sh",
    eval_script_path: str = "/eval.sh",
) -> EvalResult:
    """
    A simplified version of run_evaluation in SWE-bench.
    """
    timeout = int(timeout)
    sandbox: modal.sandbox.Sandbox | None = None
    try:
        secret = default_secret(image_url)
        image = modal.Image.from_registry(image_url, secret=secret).entrypoint([])
        app = global_modal_app()

        @MODAL_RETRY
        def create_sandbox():
            return modal.Sandbox.create(
                "sleep",
                "infinity",
                app=app,
                image=image,
                timeout=timeout,
                workdir=workdir,
                cpu=cpu,
                memory=memory,
            )

        sandbox = create_sandbox()
        assert sandbox is not None

        # write the test script
        with sandbox.open(test_script_path, "w") as f:
            f.write(test_script)

        # Write the script
        eval_script = f"cd {workdir}\nexec /bin/bash {test_script_path}"
        if setup_script is not None:
            eval_script = setup_script + "\n" + eval_script
        with sandbox.open(eval_script_path, "w") as f:
            f.write(eval_script)

        if code_patch is not None:
            with sandbox.open(code_patch_path, "w") as f:
                f.write(code_patch)

        if test_patch is not None:
            with sandbox.open(test_patch_path, "w") as f:
                f.write(test_patch)

        def apply_patch(patch_path: str) -> tuple[bool, str]:
            for git_apply_cmd in GIT_APPLY_CMDS:
                git_apply_commands = git_apply_cmd.split()
                proc = sandbox.exec(*git_apply_commands, patch_path)
                proc.wait()
                if proc.returncode == 0:
                    return True, ""
            output = proc.stdout.read()
            error = proc.stderr.read()
            message = f"{APPLY_PATCH_FAIL}:\nstdout:\n{output}\n\nstderr:{error}"
            return False, message

        if code_patch is not None:
            applied_patch, message = apply_patch(code_patch_path)
            if not applied_patch:
                return EvalResult("fail", message)

        if test_patch is not None:
            applied_patch, message = apply_patch(test_patch_path)
            if not applied_patch:
                return EvalResult("fail", message)

        run_command = f"/bin/bash {eval_script_path} 2>&1"
        start_time = time.perf_counter()
        eval_proc = sandbox.exec("bash", "-c", run_command)
        eval_proc.wait()
        test_output = eval_proc.stdout.read()
        returncode = eval_proc.poll()
        duration = time.perf_counter() - start_time

        if returncode == -1:
            outcome = "timeout"
        elif returncode == 0:
            outcome = "pass"
        else:
            outcome = "fail"

        return EvalResult(
            outcome=outcome,  # type: ignore
            message=test_output,
            duration=duration,
            returncode=returncode,
        )
    except Exception:
        trace = traceback.format_exc()
        return EvalResult(outcome="env_error", message=trace)
    finally:
        if sandbox is not None:
            sandbox.terminate()
