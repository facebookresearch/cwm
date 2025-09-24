# CWM demos

## Debugger (CLI)

With `python -m demos.cwmdbg` you can launch a pdb-style interactive debugger, backed by CWM's line-by-line code execution prediction.

To get started, spin up an fgserve instance. For example, using four GPUs:

```shell
torchrun --nproc-per-node 4 \
    -m serve.fgserve \
    gen_args.host_cache_gb=80 \
    config=serve/configs/cwm.yaml \
    checkpoint_dir=<path/to/checkpoint>
```

Then launch the debugger CLI as follows:
```shell
python -m demos.cwmdbg \
    --host <fastgen-host> \
    <path/to/checkpoint>/tokenizer.model \
    <path/to/code.py>
```

The final argument here is the code that you want to debug. For the example from our tech report (Fig B.25), the file contents would be

```py
def count_letters(s, letter):
    n = 0
    for c in s:
        n += int(c == letter)
    return n

def format_answer(word, letter, count):
    parts = [
        "Found",
        f"{count:04d}",
        "occurrences of the letter",
        letter,
        "in",
        word
    ]
    return " ".join(parts)

def f(c):  # << START_OF_TRACE
    word = "strawberry"
    num = count_letters(word, c)
    ans = format_answer(word, c, num)
    return ans
```

The CWM debugger will start at the line marked with `# << START_OF_TRACE`.
Note that the code sample above does not define a value for the parameter `c` of the function `f`, so CWM will generate a plausible value.
By using the `reset` command of the CLI app, we can re-prompt the model and obtain different values:

```
===== CONTEXT =====

<...omitted...>

===== SESSION START =====

-> def f(c):
>> print
{'c': "'a'"}
>> reset
-> def f(c):
>> print
{'c': "'r'"}
>> reset
-> def f(c):
>> print
{'c': "'t'"}
>> reset
-> def f(c):
>> print
{'c': "'s'"}
```

The commands that are implemented work as follows:

- `step`: Step to the next frame as with `step` in `pdb`. The model will recurse into function calls.
- `next`: Step to the next frame as with `next` in `pdb`. If the next frame is a function call, do not recurse into it. Instead, directly predict its return value to arrive at the next frame in the previous function scope.
- `out`: Continue frame-wise prediction until a return event of the current function is encountered. If another function is called, handle it as in `next`.
- `back`: Pop the previously predicted frame from the context.
- `print`: Print local variables as predicted.
- `reset`: Restart session.


## Debugger (notebook)

The notebook demo in `cwmdbg.ipynb` provides similar functionality as the debugger CLI application, with a graphical interface showing code alongside predicted variables.
It further allows changing variables during execution.

Usage also requires an `fgserve` instance; see the notebook for further instructions.
