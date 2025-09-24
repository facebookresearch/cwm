# Model Card for Code World Model

Code World Model (CWM) is a dense 32-billion-parameter LLM to advance research on code generation with world models.

## Model Details

### Model Description

CWM is an LLM for code generation and reasoning about code that has, in particular, been trained to better represent and reason about how code and commands affect the state of a program or system. Specifically, we mid-trained CWM on a large number of observation-action trajectories from Python execution traces and agentic interactions in containerized environments. We post-trained with extensive multi-task RL in verifiable coding, math, and multi-turn software engineering environments.

**Developed by**: Meta FAIR CodeGen Team

**Model type**: 32-billion-parameter dense decoder-only autoregressive LLM

**Language(s) (NLP)**: English

**License**: [CWM License](https://ai.meta.com/resources/models-and-libraries/cwm-license/)

**Model artifacts**:
* Post-trained model: [https://huggingface.co/facebook/cwm](https://huggingface.co/facebook/cwm)
* SFT model: [https://huggingface.co/facebook/cwm-sft](https://huggingface.co/facebook/cwm-sft)
* Pre-trained model: [https://huggingface.co/facebook/cwm-pretrain](https://huggingface.co/facebook/cwm-pretrain)
* PyTorch-checkpoints: [https://ai.meta.com/resources/models-and-libraries/cwm-downloads/](https://ai.meta.com/resources/models-and-libraries/cwm-downloads/)

**Repositories**: [https://github.com/facebookresearch/cwm](https://github.com/facebookresearch/cwm)

**Paper**: [https://ai.meta.com/research/publications/cwm/](https://ai.meta.com/research/publications/cwm/)

**GPU Requirements**: CWM can be run on a single GPU with 80 GB of VRAM with quantization.

## Uses

### Direct Use

CWM is intended for non-commercial research use in English and relevant programming languages. Relevant tasks include code synthesis and understanding.

### Out-of-Scope Use

Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English. Use in any commercial product or service. Use in any other way that is prohibited by the [FAIR Research License and Acceptable Use Policy](https://ai.meta.com/resources/models-and-libraries/cwm-license/). CWM is not intended for use as an assistant-like chat bot.

## Risks and Limitations

We explicitly release CWM as a research model under a noncommercial research license for the community to explore the opportunities afforded by world modeling and reasoning in computational environments. As such, our models come with a number of limitations which we outline below to help the research community make the most of CWM, while being aware of its shortcomings and avoiding accidental misuse.

As these are research-only models, they are not suitable for production use cases. Although we have performed some limited evaluations, we have not conducted a full range of possible evaluations for these models.  The performance of CWM in production and real-world scenarios has not been evaluated by Meta.  These models have not been fully evaluated or trained for user-facing interactions and they are not intended for such use. Researchers are recommended to exercise caution when deploying or using these models.

Similarly, CWM should not be used as a general-purpose assistant or chat model. While it was exposed to some level of instruction-following data during SFT, CWM has not undergone any thorough optimization for general chat-bot use, such as RLHF (Ouyang et al., 2022). General chat use is not an intended use of CWM and generations may diverge from expectations and/or be inappropriate or inaccurate. Further, CWM training focuses strongly on code generation and reasoning with code. Thus, our models may be lacking in other domains such as factual knowledge or classic natural language tasks.

CWM is not trained for use as a general-purpose assistant or chat model and has not been aligned on, or fully evaluated for, content risks. We make available [system level protections](https://www.llama.com/llama-protections/) – like Llama Guard, Prompt Guard and Code Shield – as a solution to help manage content generation in research environments. However, these system level protections alone are unlikely to be sufficient to enable production uses of CWM and further evaluations and fine-tuning may be required. CWM is intended to be used in English only. It is not multilingual and performance in other languages has not been evaluated or optimized.

CWM advances our commitment to providing open source technology for researchers and developers. To ensure responsible release, we have conducted an assessment of CWM’s capabilities across a range of potential threats – including plausible, catastrophic, novel, and irremediable risks – in line with Meta’s [Frontier AI Framework](https://ai.meta.com/static-resource/meta-frontier-ai-framework). The results of this assessment are detailed in the [CWM Preparedness Report](https://ai.meta.com/research/publications/cwm-preparedness). Our evaluation concludes that CWM does not significantly increase the risk of enabling threat scenarios compared to existing open source models in the ecosystem. We are committed to continuously improving our evaluation methods and welcome feedback from the community to help strengthen our approach.

## Getting Started Running CWM

There are two supported approaches to run these models:

1. With Hugging Face, where models can be used with Hugging Face [Transformers](https://github.com/huggingface/transformers) or other compatible inference engines such as [vLLM](https://github.com/vllm-project/vllm).
2. With examples from the [Code World Model](https://github.com/facebookresearch/cwm) repository. More details and extended documentation are available in the repository.


>[!IMPORTANT]
> CWM requires a dedicated system prompt to function optimally during inference. Without proper prompt configuration, CWM's output quality may be significantly degraded. The following serves as the default system prompt for reasoning tasks. For agentic workflows, append the relevant tool specifications after this base prompt. Check the [prompting guide](/PROMPTING_GUIDE.md) for more details on usage.
>
> ```
> You are a helpful AI assistant. You always reason before responding, using the following format:
>
> <think>
> your internal reasoning
> </think>
> your external response
> ```

### Hugging Face: vLLM Example

Below is a very simple example using the Hugging Face model with vLLM. Note that you can use the model with or without thinking mode.

```shell
# spawn a vLLM server
NUM_RANKS=2
vllm serve facebook/cwm --tensor-parallel-size=$NUM_RANKS

# query the model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/cwm",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant. You always reason before responding, using the following format:\n<think>\nyour internal reasoning\n</think>\nyour external response"},
      {"role": "user", "content": "Write a haiku about recursion in programming."}
    ],
    "chat_template_kwargs": {"enable_thinking": true, "preserve_previous_think": true}
  }'
```

For use with the OpenAI API, reasoning is enabled by default but can be disabled by passing `extra_body={"reasoning": {"enabled": False}}` to `OpenAI.chat.completions.create()`. When enabled, we inject `<think>\n` into the beginning of the assistant's response, forcing reasoning. This `<think>\n` is not shown to the user, but the final `</think>` generated by the model is. If disabling reasoning is desired, make sure to also use an appropriate system prompt such as "You are a helpful AI assistant.".

### CWM Repository Example (model served through Fastgen)

In the CWM code repository, we further provide examples to run inference with the PyTorch checkpoints of CWM serving the model with Fastgen, a lightweight and high-throughput inference library. See the instructions from the [README in `./serve`](./serve/README.md).

## Model architecture and training

### Architecture

CWM is an auto-regressive language model with a dense architecture composed of 64 transformer blocks. It uses an alternating pattern of local and global attention blocks interleaved in a 3:1 ratio with sliding window sizes of 8,192 and 131,072 tokens respectively. Transformer blocks use Grouped-Query Attention (GQA). CWM uses a tokenizer with a vocabulary of 128k tokens.

### Training Data

We use a variety of training data:
* Large and diverse corpora of code and English web and STEM data
* Over 30,000 executable repository Docker images
* Over 200 million memory traces of Python programs run in Docker containers
* 3 million trajectories, each consisting of simulated agentic interactions between an LLM and a computational environment
* Code and reasoning-related data such as datasets derived from GitHub pull requests similar to [SWE-RL](https://github.com/facebookresearch/swe-rl), data from compiler intermediate representations, Triton PyTorch kernels, and Lean math

### Training Procedure

CWM was trained through a standard 3-stage process: pre-training, mid-training, and post-training with both supervised fine-tuning (SFT) and reinforcement learning (RL). The model is first pre-trained on 8,192 context length for 8T tokens. It is then mid-trained at 131,072 tokens context length for an additional 5T tokens on code world modeling data. The model is then post-trained with supervised fine-tuning to improve both reasoning and general instruction following abilities. Finally, CM is trained with multi-task multi-turn verifiable reinforcement learning.

See the [Code World Model Tech Report](https://ai.meta.com/research/publications/cwm/) for additional details.

## Evaluation

Below we report results for CWM and compare to similar SOTA models on common benchmarks.


| Model                        | LCBv5   | LCBv6   | Math-500 | AIME24   | AIME25   |
|------------------------------|---------|---------|----------|----------|----------|
| Magistral-Small-2509-24B     | 70.0    | 61.6    | --       | 86.1     | 77.3     |
| Qwen3-32B                    | 65.7    | 61.9    | 97.2     | 81.4     | 72.9     |
| gpt-oss-20B (low)            | 54.2    | 47.3    | --       | 42.1     | 37.1     |
| gpt-oss-20B (med)            | 66.9    | 62.0    | --       | 80.0     | 72.1     |
| **CWM**                      | 68.6    | 63.5    | 96.6     | 76.0     | 68.2     |

| Model                           | SweBench Verified      |
|---------------------------------|------------------------|
| Devstral-1.1-2507-24B           | 53.6                   |
| Qwen3-Coder-32B                 | 51.6                   |
| gpt-oss-20B (low / med / high)* | 37.4 / 53.2 / 60.7     |
| **CWM / CWM + tts**             | 53.9 / 65.8            |

(*: GPT-5 and GPT-oss use a custom subset of 477 problems, while CWM is evaluated on the full set of 500 problems.)
