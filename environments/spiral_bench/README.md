# Spiral-Bench

Spiral-Bench is a single-turn training/evaluation environment inspired by the
public Spiral-Bench benchmark. It focuses on responses to suggestible users who
may be tempted to interpret coincidences, bodily sensations, or ordinary events
as personally significant signals.

The committed sample dataset is synthetic and generated from local templates,
so it does not copy the public benchmark prompts. Regenerate it with:

```bash
python environments/spiral_bench/generate_spiral_prompts.py \
  --num-examples 64 \
  --output environments/spiral_bench/data/spiral_bench_train_sample.jsonl
```

Each row contains `question`, `answer`, and `info` fields and can be loaded as a
Hugging Face `Dataset` by `spiral_bench.build_dataset(...)`.

The environment uses a judge rubric at scoring time. Loading the environment and
building the dataset do not call external APIs.
