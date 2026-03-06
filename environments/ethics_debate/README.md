# Ethics Debate

Multi-agent ethics debate environment: an arguer and critic take turns refining an ethical argument, scored by an LLM judge.

## How it works

1. **Arguer** presents an initial position on an ethics question
2. **Critic** identifies gaps, fallacies, and missing perspectives
3. **Arguer** refines their argument addressing the critique
4. **Critic** finds remaining weaknesses
5. **Arguer** delivers a final argument

An LLM judge scores the final argument (0-10) on thesis clarity, logical structure, counterargument handling, nuance, and depth.

## Dataset

Uses [ergotts/ethics_questions](https://huggingface.co/datasets/ergotts/ethics_questions) (2,680 ethics questions across 9 topic categories).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `judge_model` | `google/gemini-3-flash-preview` | Model for LLM judge |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API endpoint |
| `judge_api_key_var` | `PRIME_API_KEY` | Environment variable for judge API key |
| `num_rounds` | `2` | Number of arguer-critic rounds (total turns = 2*rounds + 1) |
