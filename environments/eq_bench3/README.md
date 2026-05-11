# EQ-Bench3

EQ-Bench3 is a lightweight Verifiers environment for EQ-Bench-style emotional
intensity prediction. It asks the model to infer several character emotions from
a short dialogue and scores JSON emotion-intensity predictions against synthetic
reference scores.

The committed sample dataset is generated locally from templates so it does not
copy the public EQ-Bench question set. Regenerate it with:

```bash
python environments/eq_bench3/generate_eq_bench3_prompts.py \
  --num-examples 64 \
  --output environments/eq_bench3/data/eq_bench3_train_sample.jsonl
```

Each row contains `question`, `answer`, and `info` fields and can be loaded as a
Hugging Face `Dataset` by `eq_bench3.build_dataset(...)`.
