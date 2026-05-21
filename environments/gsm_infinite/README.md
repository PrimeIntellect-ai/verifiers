# GSM-Infinite

GSM-Infinite is a single-turn train/eval environment for long-context mathematical reasoning. It loads the public `InfiniAILab/gsm_infinite_*` Hugging Face datasets and scores model completions with deterministic answer extraction.

```bash
prime env install gsm_infinite
prime eval run gsm_infinite
```

The environment exposes `load_environment(...)` with these arguments:

- `subset`: One of `symbolic`, `medium`, or `hard`. Defaults to `medium`.
- `context_length`: Dataset context length suffix such as `0`, `8k`, `16k`, `32k`, `64k`, or `128k`. Defaults to `0`.
- `train_op_splits`: Comma-separated split names or an iterable of split names. Defaults to `ops_2`.
- `eval_op_splits`: Comma-separated split names or an iterable of split names. Defaults to `ops_3`.
- `num_train_examples`: Examples per train split. Defaults to `100`; set `-1` for all rows.
- `num_eval_examples`: Examples per eval split. Defaults to `100`; set `-1` for all rows.
- `system_prompt`: Optional system prompt prepended to each example.

For `medium` and `hard`, the reward is `1.0` when the extracted integer answer matches the integer in the GSM-Infinite solution. For `symbolic`, the reward is `1.0` when the set of extracted variable names exactly matches `answer_list`.

## Citation

```bibtex
@misc{zhou2025gsminfinitellmsbehaveinfinitely,
    title={GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?},
    author={Yang Zhou and Hongyi Liu and Zhuoming Chen and Yuandong Tian and Beidi Chen},
    year={2025},
    eprint={2502.05252},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.05252},
}
```
