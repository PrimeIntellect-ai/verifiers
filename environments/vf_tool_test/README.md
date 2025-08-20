# vf-tool-test

### Overview
- **Environment ID**: `vf-tool-test`
- **Short description**: Sanity-check tool-calling environment that asks models to invoke a random subset of dummy tools.
- **Tags**: tools, single-turn, function-calling, sanity

### Datasets
- **Primary dataset(s)**: Synthetic HF dataset generated in-memory with prompts specifying required tools
- **Source links**: N/A (programmatically generated)
- **Split sizes**: Controlled by `num_train_examples` and `num_eval_examples`

### Task
- **Type**: tool use (single-turn ToolEnv)
- **Parser**: default `Parser`
- **Rubric overview**: ToolRubric checks tool execution and adds exact match on the required tool set

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-tool-test
```

Configure model and sampling:

```bash
uv run vf-eval vf-tool-test \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 1000, "num_eval_examples": 100}'
```

Notes:
- Reports are written under `./environments/vf_tool_test/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `1000` | Number of training examples |
| `num_eval_examples` | int | `100` | Number of evaluation examples |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if called tool set equals required set, else 0.0 |
| ToolRubric metrics | Tool execution success and format adherence |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<details><summary>Reports</summary>
<details><summary>vf-tool-test--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs</summary>
<p><a href="reports/vf-tool-test--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs.html" target="_blank">Open full report</a></p>
<h3>vf-tool-test: gpt-4.1-mini (n=5, r=3)</h3>
<div class="meta">
<div><b>Environment</b>: vf-tool-test (v0.1.0)</div>
<div><b>Model</b>: <span class="code">gpt-4.1-mini</span></div>
<div><b>Provider</b>: https://api.openai.com/v1</div>
<div><b>Samples</b>: n=5, r=3</div>
<div><b>Date</b>: 2025-08-08</div>
<div><b>Time</b>: 22:57:39</div>
<div><b>Sampling</b>: max_tokens=1024, temperature=0.7</div>
</div>

<h2>Reward</h2>
<table>
<tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
<tr>
<td>1.0</td>
<td>0.0</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>
</table>


<h2>Metrics</h2>
<table>
<tr>
<th>metric</th><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th>
</tr>

<tr>
<td>total_tool_calls</td>
<td>1.8</td>
<td>0.7483</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>2.0</td>
<td>2.0</td>
<td>3.0</td>
</tr>

<tr>
<td>tool_A_calls</td>
<td>0.4</td>
<td>0.4899</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

<tr>
<td>tool_B_calls</td>
<td>0.6</td>
<td>0.4899</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

<tr>
<td>tool_C_calls</td>
<td>0.2</td>
<td>0.4</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
</tr>

<tr>
<td>tool_D_calls</td>
<td>0.6</td>
<td>0.4899</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

<tr>
<td>tool_call_reward_func</td>
<td>1.0</td>
<td>0.0</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

</table>


<h2>Examples <span class="muted">(showing up to 15 of 15)</span></h2>
<div class="examples">
<table>
<tr><th>#</th><th>reward</th><th>total_tool_calls</th><th>completion</th></tr>

<tr>
<td>0</td>
<td>1.0</td>
<td>2.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>1</td>
<td>1.0</td>
<td>2.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>2</td>
<td>1.0</td>
<td>3.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>3</td>
<td>1.0</td>
<td>1.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>4</td>
<td>1.0</td>
<td>1.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>5</td>
<td>1.0</td>
<td>2.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>6</td>
<td>1.0</td>
<td>2.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>7</td>
<td>1.0</td>
<td>3.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>8</td>
<td>1.0</td>
<td>1.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>9</td>
<td>1.0</td>
<td>1.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>10</td>
<td>1.0</td>
<td>2.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>11</td>
<td>1.0</td>
<td>2.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>12</td>
<td>1.0</td>
<td>3.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>13</td>
<td>1.0</td>
<td>1.0</td>
<td><pre></pre></td>
</tr>

<tr>
<td>14</td>
<td>1.0</td>
<td>1.0</td>
<td><pre></pre></td>
</tr>

</table>
</div>
</details>
</details>
<!-- vf:end:reports -->
