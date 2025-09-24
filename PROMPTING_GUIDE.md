# Code World Model Prompting Guide

## Reserved Tokens

Reserved tokens are used for general text and chat formatting, and are not intended to be encoded from user input.
They include text sequence start and end markers, padding, chat message header delimiters, and an end of chat message token.

| String Representation | Value | Meaning |
|-----------------------|-------|---------|
|`<\|begin_of_text\|>` | 128000 | Global text sequence start marker |
|`<\|end_of_text\|>` | 128001 | Global text sequence end marker |
|`<\|pad\|>` | 128004 | Padding token |
|`<\|start_header_id\|>`| 128006 | Start of chat message header |
|`<\|end_header_id\|>` | 128007 | End of chat message header |
|`<\|eot_id\|>` | 128008 | End of chat message |


## Trace Prediction Tokens

These tokens are designed for predicting program execution traces and may be enabled when encoding user-controllable input to expose CWM's trace prediction functionality.
They include tokens for frame delimiting, action separation, function return/call, next line, exception, and argument separation, as well as a sentinel token for the start of the source code context for trace prediction (see below).

| String Representation | Value | Meaning |
|-----------------------|-------|---------|
|`<\|frame_sep\|>` | 128100 | Start of trace sentinel, end of execution step |
|`<\|action_sep\|>` | 128101 | Start of source code line |
|`<\|return_sep\|>` | 128102 | Execution step: return from function scope |
|`<\|call_sep\|>` | 128103 | Execution step: enter function scope |
|`<\|line_sep\|>` | 128104 | Execution step: next line |
|`<\|exception_sep\|>` | 128105 | Execution step: exception |
|`<\|arg_sep\|>` | 128106 | Separator for return and exception values |
|`<\|trace_context_start\|>` | 128107 | Start of source code context for trace prediction |


## Chat Format

A chat is structured as a list of messages, each with the following format:
```
<|start_header_id|>$ROLE<|end_header_id|>

$CONTENT<|eot_id|>
```

The `$ROLE` can be `system`, `user`, `assistant`, or `tool: $TOOL`. `$CONTENT` is the message content.
The model is to be prompted with an assistant header followed by two newline characters; the `<|eot_id|>` token marks the end of its reply and is thus the stop token for inference.
The conversation's first token is expected to be `<|begin_of_text|>`.


## Reasoning

CWM is a hybrid reasoning and non-reasoning model; reasoning mode is enabled via prompting. Reasoning mode is turned on by starting the system prompt with:
```
You are a helpful AI assistant. You always reason before responding, using the following format:

<think>
your internal reasoning
</think>
your external response
```
The model should be prompted with a leading `<think>\n`, i.e., a prompt should end with (showing newline characters for clarity here):
```
<|start_header_id|>assistant<|end_header_id|>\n\n<think>\n
```
The reasoning section will be closed with `</think>`, and any text produced afterwards is the answer to the preceding user input.


## Tool Use

The model performs tool calls with the following format:
```
<tool: $TOOL>
$CONTENT
</tool>
```
Any available tools are to be announced in the system prompt. User code is responsible for detecting tool calls in model output and responding with a message marked with the respective role.

An example tool output of the `python` tool could be:
```
<|start_header_id|>tool: python<|end_header_id|>

completed.
[stdout]$STDOUT_CONTENT[/stdout]
[stderr]$STDERR_CONTENT[/stderr]<|eot_id|>
```
Control is then handed back to the model for further processing.

An example of how tools can be specified in the system prompt:
```
You have access to the following tools:

<tool: bash>
[command(s)]
</tool>
Executes bash command(s) [command(s)] in the current session. [command(s)] can be any non-interactive bash command(s) either single or multi-line.

<tool: create>
[path]
[content]
</tool>
Creates a new file at [path] with [content], where [path] must not exist, but its parent directory must exist.
```
Here, the model may invoke either the `bash` or the `create` tool.


## Trace Prediction

CWM is able to predict the execution of Python programs on a step-by-step basis using dedicated trace prediction tokens. The prompt requires a source code context, `$CONTEXT`, and a sentinel `<|frame\_sep|>` token to induce trace prediction, structured as:
```
<|begin_of_text|><|trace_context_start|>$CONTEXT<|frame_sep|>
```
In `$CONTEXT`, the entry point for trace prediction is marked with a `<< START_OF_TRACE` comment. An execution trace in CWM is a series of *frames*, with each frame consisting of an *observation* (local variables) and an *action* (source code line). There are four different types of frames, formatted as follows:
```
<|call_sep|>$LOCALS<|action_sep|>$SOURCE<|frame_sep|>
<|line_sep|>$LOCALS<|action_sep|>$SOURCE<|frame_sep|>
<|return_sep|><|action_sep|>$SOURCE<|arg_sep|>$VALUE<|frame_sep|>
<|exception_sep|><|action_sep|>$SOURCE<|arg_sep|>$VALUE<|frame_sep|>
```
The model produces an `<|end_of_text|>` token to denote the end of the execution, which is reached when exiting the scope of the trace's entry point. Locals are formatted as JSON key-value pairs where values are always rendered as JSON strings. The `$VALUE` for return and exception frames is also a JSON-encoded string representation.
