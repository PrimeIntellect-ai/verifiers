# IFBench

IFBench is a single-turn environment for precise instruction-following RLVR. It loads the `allenai/IFBench_test` dataset and scores completions with the official IFBench instruction checkers vendored from AllenAI's Apache-2.0 implementation.

```bash
prime env install ifbench
prime eval run ifbench
```

The environment exposes `load_environment(...)` with these arguments:

- `dataset_name`: Hugging Face dataset name. Defaults to `allenai/IFBench_test`.
- `dataset_split`: Dataset split. Defaults to `train`.
- `num_train_examples`: Number of training examples to load. Defaults to all examples.
- `num_eval_examples`: Number of evaluation examples to load. Defaults to all examples.
- `reward_mode`: `loose` or `strict`, matching IFBench's official evaluation modes. Defaults to `loose`, the headline metric used in the IFBench paper.
- `system_prompt`: Optional system prompt prepended to each example.

Rows are converted into standard Verifiers single-turn prompts, while each row's IFBench instruction IDs and checker kwargs are kept in `info` for deterministic reward computation.

## Citation

This environment vendors the IFBench evaluator released by AllenAI:

```bibtex
@misc{pyatkin2025generalizing,
   title={Generalizing Verifiable Instruction Following},
   author={Valentina Pyatkin and Saumya Malik and Victoria Graf and Hamish Ivison and Shengyi Huang and Pradeep Dasigi and Nathan Lambert and Hannaneh Hajishirzi},
   year={2025},
   journal={Advances in Neural Information Processing Systems},
   volume={38},
   year={2025}
}
```
