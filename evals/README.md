# CWM evaluation code

We share here the code to reproduce key results from the CWM technical report, including the SWE-bench Verified (SBV) and LiveCodeBench (LCB) evaluations and math benchmark results.


## Evaluation

### SWE-bench Verified (SBV)

> Environment: [cwm/rl/envs/envs/swerl_tool.py](/cwm/rl/envs/envs/swerl_tool.py)

First, make sure you have set up [Modal](https://modal.com/docs/guide). Then, follow the steps below to run agentic inference and eval on SBV (one rollout per problem). Note that there can be a 2-3 point variance in the final score due to randomness. Please also check your modal account usage to make sure you do not hit the rate limits; otherwise, the evals will fail and the score will be lower.

```bash
# 1. Create SBV eval file
python -m evals.scripts.swerl.format_swerl \
  save_path=swebench_verified_swerl.jsonl \
  dataset=princeton-nlp/SWE-bench_Verified \
  split=test \
  namespace=swebench

# 2. Create the concrete config
sed "s/{{eval_data_path}}/swebench_verified_swerl.jsonl/g" evals/configs/eval_sbv.yaml > eval_sbv_base_config.yaml

# 3. Run agentic inference plus evaluation
python -m torch.distributed.run --nproc_per_node=8 -m evals.main -- \
  config=eval_sbv_base_config.yaml \
  dump_dir=eval-cwm-sbv \
  checkpoint_dir=</path/to/cwm/checkpoint>
```

The structure of the output folder `eval-cwm-sbv` is as follows.

```
❯ tree eval-cwm-sbv
eval-cwm-sbv
├── rl_eval_config.yaml
├── rl_eval.log
└── trajectories
    ├── all_metrics.jsonl
    └── swerl_tool_envthink_tagmodalevalsimilaritiesswebench_verified_swerl_jsonl
        ├── swerl_tool_envthink_tagmodalevalsimilaritiesswebench_verified_swerl_jsonl.jsonl
        └── task_config.yaml
```

With 8 H100 GPUs, the eval takes about 6 hours. After the eval is done, you can find a `metrics.rl_eval.jsonl` file that contains the final score in the `eval-cwm-sbv` folder. If you want to know the intermediate scores, you can run the following command. Typically, the score will be higher at the beginning because trajectories that finish earlier are dumped first, and they tend to be easier problems.

```bash
❯ jq -s 'map(select(.metrics[0].terminal_metrics.pass == true)) | length' eval-cwm-sbv/trajectories/all_metrics.jsonl \
| awk -v total=$(wc -l < eval-cwm-sbv/trajectories/all_metrics.jsonl) '{print $1 " / " total}'
26 / 36
```

#### Test-time scaling (TTS)

> Environment: [cwm/rl/envs/envs/swerl_testgen_tool.py](/cwm/rl/envs/envs/swerl_testgen_tool.py)

Our testing-based TTS involves several steps (more details in the paper):
1. Generate 40 samples per problem with `samples_per_prompt: 40` with the config at [evals/configs/eval_sbv.yaml](./configs/eval_sbv.yaml) and run the eval as above.
2. Generate 40 test samples per problem with `samples_per_prompt: 40` with the config at [evals/configs/testgen_sbv.yaml](./configs/testgen_sbv.yaml) and run the eval as above.
3. For each sample, run against existing tests, following [Agentless](https://github.com/OpenAutoCoder/Agentless). Note, we use the raw existing passing tests without additional filtering.
4. For each sample, run against top-5 majority tests (after filtering) generated in step 2.
5. Pick the sample that passes the most tests, break ties by picking the majority one, and if still ties, pick the one with fewest tokens.

Step 3-5 depends on our own fork of the [Agentless](https://github.com/OpenAutoCoder/Agentless) repo for test execution. Due to the usage of internal infrastructure, we cannot open-source this part of the code. However, you should be able to replicate this part with some minor adjustments of the original Agentless repo.

We also provide a simple majority-voting TTS script that can be run after the above eval is done. In this case, you should set `samples_per_prompt` to some $k$, and then run [evals/scripts/swerl/tts_majority_voting.py](./scripts/swerl/tts_majority_voting.py) to get the score.

#### Eval utility

We provide a fast eval CLI script based on Modal (5 min for SWE-bench Verified eval with enough threads): [evals/scripts/swerl/eval_cli.py](./scripts/swerl/eval_cli.py). It can be used to eval any eval file in SWE-bench `all_preds.jsonl` format. For more details, check the script's docstring.

### LiveCodeBench (LCB)

> Environment: [cwm/rl/envs/envs/lcb.py](/cwm/rl/envs/envs/lcb.py)

The steps below describe how to run inference and eval on LCBv5 (10/1/2024 to 2/1/2025) or LCBv6 (8/1/2024 to 5/1/2025) (10 rollouts per problem). They are similar to the SBV eval steps above, and the output folder follows the same structure.

```bash
# 1. Download LCBv5 and filter by date range
python -m evals.scripts.download_lcb \
  save_path=LCB.jsonl \
  version=v5
# for LCBv6 set version=v6

# 2. Create the concrete config
sed "s/{{eval_data_path}}/LCB.jsonl/g" evals/configs/eval_lcb.yaml > eval_lcb_base_config.yaml

# 3. Run inference plus evaluation
python -m torch.distributed.run --nproc_per_node=8 -m evals.main -- \
  config=eval_lcb_base_config.yaml \
  dump_dir=eval-cwm-lcb \
  checkpoint_dir=</path/to/cwm/checkpoint>
```

With 8 H100 GPUs, the eval takes about 4 hours.

### Math (MATH-500, AIME24, AIME25)

> Environment: [cwm/rl/envs/envs/math_think_dialog.py](/cwm/rl/envs/envs/math_think_dialog.py)

The steps below describe how to run inference and eval on MATH-500, AIME24, or AIME25 (20 rollouts per problem). They are similar to the  other evals, and the output folder follows the same structure.

```bash
# 1. Download MATH-500
python -m evals.scripts.download_math \
  save_path=math.jsonl \
  dataset=HuggingFaceH4/MATH-500
# for AIME24 set dataset=HuggingFaceH4/aime_2024
# for AIME25 set dataset=yentinglin/aime_2025

# 2. Create the concrete config
sed "s/{{eval_data_path}}/math.jsonl/g" evals/configs/eval_math.yaml > eval_math_base_config.yaml

# 3. Run inference plus evaluation
python -m torch.distributed.run --nproc_per_node=8 -m evals.main -- \
  config=eval_math_base_config.yaml \
  dump_dir=eval-cwm-math \
  checkpoint_dir=</path/to/cwm/checkpoint>
```

With 8 H100 GPUs, the eval takes about 5 hours for MATH-500 and 2 hours for AIME24/25.
