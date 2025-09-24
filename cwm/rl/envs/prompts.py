# Copyright (c) Meta Platforms, Inc. and affiliates.


THINK_TAG_START = "<think>\n"
THINK_TAG_END = "</think>"
CWM_THINKTAG_SYS_PROMPT = """
{description} You always reason before responding, using the following format:

<think>
your internal reasoning
</think>
your external response
""".strip()

CWM_THINKTOKEN_SYS_PROMPT = "{description} You always reason before responding."

ENV_PROMPT = """
# Environment

{environment}
"""


def get_cwm_sys_prompt(
    think: bool,
    # think = True & think_tag = False means using think token
    use_think_tag: bool,
    description: str = "You are a helpful AI assistant.",
    # tool specs are part of `environment`
    environment: str | None = None,
) -> str:
    assert len(description) > 0, "Description must not be empty."
    if use_think_tag:
        assert think, "If using think tag, reasoning mode must be think."

    if environment is not None:
        environment = environment.strip()
        assert len(environment) > 0, "Environment must not be empty."

    if think and use_think_tag:
        prompt_template = CWM_THINKTAG_SYS_PROMPT
    elif think and not use_think_tag:
        prompt_template = CWM_THINKTOKEN_SYS_PROMPT
    elif not think:
        prompt_template = "{description}"
    prompt = prompt_template.format(description=description).strip()
    if environment:
        env_prompt = ENV_PROMPT.format(environment=environment).strip()
        prompt = "\n\n".join([prompt, env_prompt]).strip()
    return prompt


def think_tag_prompt() -> str:
    return THINK_TAG_START
