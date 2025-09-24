# Code World Model

Code World Model (CWM) is a 32-billion-parameter open-weights LLM, to advance research on code generation with world models.

The CWM release includes model weights (pre-trained, SFT and instruction-tuned), the technical report, the model card and starting code to run inference with the model, and reproduce the reported numbers on key benchmarks such as SWE-bench Verified, LiveCodeBench, AIME and MATH.

Check out the accompanying [Code World Model Tech Report](https://ai.meta.com/research/publications/cwm/).

## Summary

We introduce Code World Model (CWM). CWM is an LLM for code generation and reasoning about code that has, in particular, been trained to better represent and reason about how code and commands affect the state of a program or system. Specifically, we mid-trained CWM on a large number of observation-action trajectories from Python execution traces and agentic interactions in containerized environments. We post-trained with extensive multi-task RL in verifiable coding, math, and multi-turn software engineering environments.

## Model download

### Hugging Face weights

The model weights are available on Hugging Face for use with vLLM.

| Model | Description | Download |
|---|---|---|
| `cwm` | Instruction-tuned model |  [🤗 Hugging Face](https://huggingface.co/facebook/cwm) |
| `cwm-sft` | SFT model | [🤗 Hugging Face](https://huggingface.co/facebook/cwm-sft) |
| `cwm-pretrain` | Pre-trained model | [🤗 Hugging Face](https://huggingface.co/facebook/cwm-pretrain) |

To download the weights from Hugging Face, please follow these steps:
* Visit one of the repos, for example [facebook/cwm](https://huggingface.co/facebook/cwm)
* Read and accept the license. Once your request is approved, you'll be granted access to all the CWM model weights. Note that requests used to take up to one hour to get processed.

### PyTorch weights

We further distribute [PyTorch checkpoints](https://ai.meta.com/resources/models-and-libraries/cwm-downloads) in PyTorch Distributed Checkpoint format (DCP) for developers looking to dive deeper or use the code released in this repository. Once your request is approved, you will receive a signed URL over email which can be used with `./download_pytorch.sh <URL>"`. The links may expire after 24 hours or a fixed number of downloads. Please re-request if you start seeing errors such as `403: Forbidden`.

Please refer to the model card below for model details, more information on uses, risks and limitations as well as the license under which the weights are released.

## Model card

See the [model card](./MODEL_CARD.md).

## Getting started within the CWM repository

### Environment setup

```shell
micromamba env create -f environment.yaml -n CWM
```
Note: Use a recent version of micromamba (>= 2.2.0) to ensure that the environment variables are set correctly.

160GB of combined GPU VRAM (e.g., two Nvidia H100 GPUs) and RDMA (Mellanox 5 InfiniBand or AWS EFA) are required to run evaluations and demos in this repository with their respective default configurations.

### Tips on model use

>[!IMPORTANT]
> CWM requires a dedicated system prompt to function optimally during inference. Without proper prompt configuration, CWM's output quality may be significantly degraded. Check [MODEL_CARD.md](./MODEL_CARD.md) for details.

### Running inference with CWM

For a reference inference implementation, we provide a simple serving endpoint to locally run inference with the CWM model from the PyTorch DCP weights and a Fastgen (Carbonneaux et al.) server. Checkout the [serve README](./serve/README.md) to get started.

### Reproducing evaluation results

We provide the main logic to run inference and reproduce our agentic and reasoning evaluation results with CWM.
Check out the [evals README](./evals/README.md) for instructions.

### Demos

We provide demos to showcase the model capabilities. With our demos, you can use CWM as a neural debugger. Get started with the [demos README](./demos/README.md).

## Examples with Hugging Face weights

For example and starting code to use CWM from the Hugging Face weights, please refer to the [model card](./MODEL_CARD.md).

## License

* The code in this repository is released under a BSD-3 license as found in the [LICENSE file](./LICENSE).
* The model weights are released under a custom license as found on the [CWM License page](https://ai.meta.com/resources/models-and-libraries/cwm-license/).

## Citation

```
@misc{cwm2025,
  author       = {FAIR CodeGen Team, Meta},
  title        = {CWM: An Open-Weights LLM for Research on Code Generation with World Models},
  year         = {2025},
  url          = {https://ai.meta.com/research/publications/cwm/}
}
```
