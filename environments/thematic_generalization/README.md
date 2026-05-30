# Thematic Generalization

Single-turn Verifiers environment for the [LLM Thematic Generalization Benchmark](https://github.com/lechmazur/generalization).

Each task gives the model:

- 3 positive examples of a narrow latent theme
- 3 misleading anti-examples
- 8 candidate matches

The model scores every candidate from 0 to 10 using the requested XML tags. The reward ranks the hidden correct candidate by the model's scores and returns reciprocal rank, so placing the correct candidate first receives `1.0`.

## Usage

```bash
prime env install thematic-generalization
prime eval run thematic-generalization -n 5 -r 1
```

## Dataset

The bundled JSONL files are derived from `lechmazur/generalization` V1 pick prompts. The source `<<LEFTOVER>>` marker is removed from model-visible prompts and stored as the answer label.

- `train.jsonl`: 648 examples
- `eval.jsonl`: 162 examples
