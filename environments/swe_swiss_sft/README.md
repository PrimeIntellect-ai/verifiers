# SWE-Swiss SFT

SWE-Swiss SFT is a train/eval environment over the public SWE-Swiss supervised
fine-tuning datasets. It converts each row's chat transcript into a single-turn
code-repair prompt with the first assistant response as the target answer.

By default the environment uses `SWE-Swiss/SWESwiss-SFT-Repair-4K`:

```bash
prime eval run swe-swiss-sft \
  -a '{"num_eval_examples": 20}'
```

Useful arguments:

| Arg | Default | Description |
| --- | --- | --- |
| `dataset_name` | `SWE-Swiss/SWESwiss-SFT-Repair-4K` | Hugging Face dataset to use for training |
| `eval_dataset_name` | same as `dataset_name` | Optional separate eval dataset |
| `split` | `train` | Training split |
| `eval_split` | same as `split` | Evaluation split |
| `num_train_examples` | `-1` | Limit train examples |
| `num_eval_examples` | `-1` | Limit eval examples |
| `shuffle_seed` | `777` | Deterministic shuffle seed, or `null` to preserve source order |

The reward is normalized text similarity against the reference assistant
response. This keeps the environment lightweight and usable for the existing
SWE-Swiss SFT data without requiring a sandboxed repository checkout.
