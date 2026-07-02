# Evaluation

To evaluate any environment, use the `eval` entrypoint:

```bash
prime eval primeintellect/terminal-bench-2
```

You can also use `.toml` files for configuration:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0

[taskset]
id = "primeintellect/terminal-bench-2"

[harness]
id = "codex"
version = "0.116.0"
```

Validate the config by using `prime eval @ config.toml --dry-run`. To run the evaluation, use `prime eval @ config.toml` to run it locally or `prime eval @ config.toml --hosted` to use [hosted evaluations](https://docs.primeintellect.ai/tutorials-environments/hosted-evaluations).

Use dotted arguments to set values using the CLI, e.g. `--sampling.temperature 0.5`. CLI arguments overwrite toml arguments when both are present.

The output from evaluations are written into `outputs/<taskset>--<model>--<harness>/<uuid>/` by default (use `output_dir` to overwrite the folder). The folder contains the used `config.toml`, all the traces in `results.jsonl`, as well as logs of the run and workers in `eval.log`.
