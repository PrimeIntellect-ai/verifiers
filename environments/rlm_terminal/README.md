# rlm_terminal

Runs the experimental composable RLM harness on Terminal-Lego tasks.

The taskset expects Terminal-Lego task directories and prebuilt per-task images.
By default this package uses the included image map for images already pushed to
our registry. Pass `dataset_path` or set `TERMINAL_LEGO_DATASET_PATH` to a local
Git LFS clone of `SWE-Lego/Terminal-Lego-15k`.

Example:

```bash
uv run vf-eval rlm_terminal --env-dir-path environments \
  -p prime -m openai/gpt-5.5 -n 1 -r 1 -c 1 -d -v --max-retries 0 \
  -a '{"dataset_path":"/path/to/Terminal-Lego-15k","task_names":["task_00000"]}'
```
