# GSM-Infinite

GSM-Infinite is a train/eval environment for Infini-AI-Lab's scalable
long-context math reasoning benchmark. It loads the public Hugging Face datasets
from the GSM-Infinite collection and normalizes rows into Verifiers
`question`/`answer`/`info` examples.

By default the environment uses the medium zero-context dataset and the
`ops_2` split:

```bash
prime eval run gsm-infinite \
  -a '{"subset": "medium", "context_length": "0", "operations": 2, "num_eval_examples": 20}'
```

Useful arguments:

| Arg | Default | Description |
| --- | --- | --- |
| `subset` | `medium` | One of `symbolic`, `medium`, or `hard` |
| `context_length` | `0` | Dataset suffix length, e.g. `0`, `8k`, `16k`, `32k` |
| `operations` | `2` | Operation split loaded as `ops_<operations>` |
| `num_train_examples` | `-1` | Limit train examples |
| `num_eval_examples` | `-1` | Limit eval examples |

The reward compares the model's final `Answer: ...` value to the normalized
dataset answer. The environment does not require provider APIs during dataset
loading or scoring.
