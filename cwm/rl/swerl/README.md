# CWM SWE-RL

This module provides a tool-based execution environment for **evaluating** LLM agents on software engineering tasks. SWE-RL enables agents to interact with containerized environments using structured tool calls to solve issues, edit code, and implement features.

## Code

The SWE-RL codebase is organized around a core tool execution framework with pluggable backends for container management.

[`tools.py`](tools.py) defines the `ToolBackend` protocol and core abstractions for tool execution. This includes:
- `ToolBackend`: Protocol for container-based tool execution
- `ToolCallResult`: Result dataclass for tool operations
- Tool builders like `make_bash`, `make_python_plugin`, and submission helpers

[`errors.py`](errors.py) defines SWE-RL-specific exceptions:
- `BackendInitError`: Container initialization failures
- `FormatError`: Tool call parsing errors
- `NoSuchToolError`: Invalid tool references

Below are the core components of the SWE-RL framework.

### Remote Execution

[`remote/`](remote/) implements the client-server architecture for bash-like communication. It shares the insight with [SWE-ReX](https://github.com/SWE-agent/SWE-ReX) but is designed to be lightweight and easier to deploy:
- [`session.py`](remote/session.py): `AsyncSession` class for persistent bash-like sessions
- [`client.py`](remote/client.py): Client interface for remote command execution
- [`server.py`](remote/server.py): Server implementation for handling remote requests

These components are supposed to be **deployed inside the container** and launched with `python`, so the code is backward-compatible by design with an older version of Python.

An example of running the remote server and client:

```bash
python -m cwm.rl.swerl.remote.server --host 0.0.0.0 --port 8888

# In another terminal, run the client:
echo '{"method": "get_new_id"}' | python -m cwm.rl.swerl.remote.client_cli --host localhost --port 8888
# Server will produce logs and client will receive a response
```

### Container Backends

[`modal_backend.py`](modal_backend.py) provides [Modal](https://modal.com/)-based containerized execution with cloud infrastructure.

[`default_backends.py`](default_backends.py) contains factory functions for creating preconfigured backend instances with standard tool sets. An example below with Modal (make sure you have followed its [setup instructions](https://modal.com/docs/guide) and updated to the latest builder version):

```python
from cwm.rl.swerl.default_backends import get_default_modal_backend

backend = get_default_modal_backend(
    image_url="python:3.11-slim",
    session_timeout=120.0,
    work_dir="/",
    use_tunnel=False,
    background_mode=False,
    server_python_path="python",
)

with backend:
    result = backend.apply_tool("bash", "echo 'Hello, SWE-RL!'")
    print(result.output)
    # Hello, SWE-RL!
```

### Tools and Plugins

[`default_tools.py`](default_tools.py) defines the core tool interface:
- `DEFAULT_TOOLS`: Standard tool set including `bash` and `submit`
- `parse_tool_calls()`: Parses tool calls from agent responses using regex pattern `<tool: name>content</tool>`

[`plugins/`](plugins/) are *syntax sugars* of the bash tool. They're python scripts that receives input from the agent and produce a specific output. Plugin calls are desugared into bash commands like so:

```bash
${{PYPLUGIN_PYTHON_PATH:-python3}} {script_path} <<'{EOF}'
{command}
{EOF}
```

- [`edit.py`](plugins/edit.py): File editing with search/replace using `<<<<<<< SEARCH` / `>>>>>>> REPLACE` syntax
- [`create.py`](plugins/create.py): File creation tool

### Evaluation

[`eval_backend/`](eval_backend/) contains evaluation infrastructure:
- [`eval.py`](eval_backend/eval.py): Core evaluation logic for SWE-bench style tasks
- [`utils/`](eval_backend/utils/): Evaluation utilities and constants

[`similarities.py`](similarities.py), adapted from the [SWE-RL project](https://github.com/facebookresearch/swe-rl) implements patch similarity metrics using unified diff comparison for reward computation.

### Configuration

[`default_configs.py`](default_configs.py) provides environment constants:
- Memory and thread limits for containers
- Session startup scripts
- Plugin configuration paths
- Default Python interpreter paths

[`common.py`](common.py) contains utilities for loading configurations from CLI with support for nested frozen dataclasses.

## Tool Interface

Agents interact with the environment through structured tool calls:

```
<tool: bash>
ls -la src/
</tool>

<tool: edit>
src/example.py
<<<<<<< SEARCH
def old_function():
    return "old"
=======
def new_function():
    return "new"
>>>>>>> REPLACE
</tool>

<tool: create>
tests/test_new.py
import unittest

class TestNew(unittest.TestCase):
    def test_example(self):
        self.assertTrue(True)
</tool>

<tool: submit>
fix.patch
</tool>
```

## Environment Integration

SWE-RL integrates with the broader CWM RL framework through [`cwm/rl/envs/envs/swerl_tool.py`](../envs/envs/swerl_tool.py), which defines:

- `SWERLConfig`: Configuration dataclass for environment parameters
- `SWERLToolEnv`: Main environment class implementing the tool-based interaction paradigm
- System prompts for issue solving tasks with detailed tool specifications

## Development Notes

SWE-RL is designed to serve as a lightweight but general framework for an agent solving tasks inside a containerized environment with a persistent session and basic editing tools. This includes, but not limited to, solving software issues, creating new unit tests, and doing competitive programming in a terminal. There are several ways to extend the framework even further:

- Adding new tools for specialized tasks like code analysis, linting, or testing
- Implementing new plugins for specific tasks, such as a new editing tool or domain-specific plugins
- Adding new container backends for different cloud providers or local execution environments (e.g., Docker, Kubernetes, etc.)
- Expanding the evaluation framework to support more types of validation and testing scenarios
