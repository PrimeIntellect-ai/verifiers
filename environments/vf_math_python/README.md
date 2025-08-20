# vf-math-python

### Overview
- **Environment ID**: `vf-math-python`
- **Short description**: Tool-using math environment requiring Python tool calls to compute answers; graded by symbolic equivalence.
- **Tags**: math, tools, python, single-turn, boxed-answer

### Datasets
- **Primary dataset(s)**: Example `math` dataset via `load_example_dataset`
- **Source links**: Uses example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via args; defaults to `train` split and all examples

### Task
- **Type**: tool use (single-turn ToolEnv)
- **Parser**: Basic `Parser` with boxed answer extraction
- **Rubric overview**: Correctness by `math_verify.parse` + `verify`; logs auxiliary metrics (#turns, #tool calls, #errors)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-math-python
```

Configure model and sampling:

```bash
uv run vf-eval vf-math-python \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"dataset_name": "math", "dataset_split": "train", "num_train_examples": -1}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports are written under `./environments/vf_math_python/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"math"` | Example dataset to load |
| `dataset_split` | str | `"train"` | Split to load |
| `num_train_examples` | int | `-1` | Limit dataset size (`-1` for all) |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `correct_answer_reward_func` | 1.0 if symbolic verification passes, else 0.0 |
| `num_turns` | Number of assistant messages in completion |
| `num_tool_calls` | Number of tool messages in completion |
| `num_errors` | Count of tool error messages |

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<details><summary>Reports</summary>
<details><summary>vf-math-python--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs</summary>
<p><a href="reports/vf-math-python--v0.1.0--model=gpt-4.1-mini--n=5--r=3--args=noargs.html" target="_blank">Open full report</a></p>
<h3>vf-math-python: gpt-4.1-mini (n=5, r=3)</h3>
<div class="meta">
<div><b>Environment</b>: vf-math-python (v0.1.0)</div>
<div><b>Model</b>: <span class="code">gpt-4.1-mini</span></div>
<div><b>Provider</b>: https://api.openai.com/v1</div>
<div><b>Samples</b>: n=5, r=3</div>
<div><b>Date</b>: 2025-08-08</div>
<div><b>Time</b>: 17:25:56</div>
<div><b>Sampling</b>: max_tokens=1024, temperature=0.7</div>
</div>

<h2>Reward</h2>
<table>
<tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
<tr>
<td>0.9933</td>
<td>0.0249</td>
<td>15</td>
<td>0.97</td>
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
<td>correct_answer_reward_func</td>
<td>1.0</td>
<td>0.0</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

<tr>
<td>num_turns</td>
<td>1.6</td>
<td>0.4899</td>
<td>15</td>
<td>1.0</td>
<td>1.0</td>
<td>2.0</td>
<td>2.0</td>
<td>2.0</td>
</tr>

<tr>
<td>num_tool_calls</td>
<td>0.7333</td>
<td>0.6799</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>1.0</td>
<td>1.0</td>
<td>2.0</td>
</tr>

<tr>
<td>num_errors</td>
<td>0.0667</td>
<td>0.2494</td>
<td>15</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.3</td>
</tr>

</table>


<h2>Examples <span class="muted">(showing up to 15 of 15)</span></h2>
<div class="examples">
<table>
<tr><th>#</th><th>reward</th><th>correct_answer_reward_func</th><th>completion</th></tr>

<tr>
<td>0</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The values of \(a\) and \(b\) that make the piecewise function continuous are \(a = -3\) and \(b = 3\). The sum \(a + b\) is:

\[
a + b = -3 + 3 = 0
\]

So, \(a + b = 0\).</pre></td>
</tr>

<tr>
<td>1</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Let&#39;s denote:
- \( r \) = the original number of rows,
- \( m \) = the original number of members in each row,
- \( N \) = the total number of band members.

From the problem, we have the following conditions:

1. The total number of band members is less than 100:
   \[
   N &lt; 100
   \]

2. When ar…</pre></td>
</tr>

<tr>
<td>2</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial given is:

\[ 4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9 \]

Let&#39;s identify the degree by analyzing the powers of \(x\) in each term.

- \(4\) is a constant term (degree 0).
- \(5x^3\) is a term with degree 3.
- \(100\) is a constant term (degree 0).
- \(2\pi x^4\) is a term with …</pre></td>
</tr>

<tr>
<td>3</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The value of \(\left\lceil3\left(6-\frac12\right)\right\rceil\) is 17.</pre></td>
</tr>

<tr>
<td>4</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Sam did not work for 6 days.</pre></td>
</tr>

<tr>
<td>5</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The values of \( a \) and \( b \) that make the function continuous are:
\[
a = -3, \quad b = 3.
\]

Therefore,
\[
a + b = -3 + 3 = 0.
\]

The value of \( a + b \) is \( \boxed{0} \).</pre></td>
</tr>

<tr>
<td>6</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Let&#39;s denote:

- \( m \) = number of members in each row in the original formation,
- \( r \) = number of rows in the original formation,
- \( N \) = total number of band members.

From the problem:

1. \( N &lt; 100 \).
2. The original formation is \( m \times r \) but with 2 members left over, so:
 …</pre></td>
</tr>

<tr>
<td>7</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial given is:

\[4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9\]

Let&#39;s identify the degree of each term:
- \(4\) is a constant term (degree 0).
- \(5x^3\) has degree 3.
- \(100\) is a constant term (degree 0).
- \(2\pi x^4\) has degree 4.
- \(\sqrt{10} x^4\) has degree 4.
- \(9\) is a c…</pre></td>
</tr>

<tr>
<td>8</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The value of \(\left\lceil3\left(6-\frac12\right)\right\rceil\) is 17.</pre></td>
</tr>

<tr>
<td>9</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Sam did not work for 6 days.</pre></td>
</tr>

<tr>
<td>10</td>
<td>0.9</td>
<td>1.0</td>
<td><pre>From the first calculation, we have the values:
\[
a = -3, \quad b = 3
\]

Now, let&#39;s find \( a + b \):
\[
a + b = -3 + 3 = 0
\]

So, the value of \( a + b \) is 0.</pre></td>
</tr>

<tr>
<td>11</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Let the number of members in each of the \( r \) rows be \( m \). Then the total number of band members \( N \) satisfies:
\[
N = mr + 2
\]
since there are 2 members left over when arranged in \( r \) rows of \( m \) members.

If the director increases the number of members per row by 1 (i.e., \( m…</pre></td>
</tr>

<tr>
<td>12</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial is:

\[4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9\]

Let&#39;s identify the degree by the highest power of \(x\) with a nonzero coefficient.

The terms and their degrees are:
- \(4\) (degree 0)
- \(5x^3\) (degree 3)
- \(100\) (degree 0)
- \(2\pi x^4\) (degree 4)
- \(\sqrt{10} x^4\) (d…</pre></td>
</tr>

<tr>
<td>13</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The value of \(\left\lceil3\left(6-\frac{1}{2}\right)\right\rceil\) is \(\boxed{17}\).</pre></td>
</tr>

<tr>
<td>14</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Sam did not work for 6 days.</pre></td>
</tr>

</table>
</div>
</details>
<details><summary>vf-math-python--v0.1.0--model=gpt-4.1-mini--n=3--r=2--args=noargs</summary>
<p><a href="reports/vf-math-python--v0.1.0--model=gpt-4.1-mini--n=3--r=2--args=noargs.html" target="_blank">Open full report</a></p>
<h3>vf-math-python: gpt-4.1-mini (n=3, r=2)</h3>
<div class="meta">
<div><b>Environment</b>: vf-math-python (v0.1.0)</div>
<div><b>Model</b>: <span class="code">gpt-4.1-mini</span></div>
<div><b>Provider</b>: https://api.openai.com/v1</div>
<div><b>Samples</b>: n=3, r=2</div>
<div><b>Date</b>: 2025-08-08</div>
<div><b>Time</b>: 16:57:08</div>
<div><b>Sampling</b>: max_tokens=1024, temperature=0.7</div>
</div>

<h2>Reward</h2>
<table>
<tr><th>mean</th><th>std</th><th>n</th><th>p5</th><th>p25</th><th>p50</th><th>p75</th><th>p95</th></tr>
<tr>
<td>0.8167</td>
<td>0.367</td>
<td>6</td>
<td>0.225</td>
<td>0.925</td>
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
<td>correct_answer_reward_func</td>
<td>0.8333</td>
<td>0.3727</td>
<td>6</td>
<td>0.25</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
</tr>

<tr>
<td>num_turns</td>
<td>1.5</td>
<td>0.7638</td>
<td>6</td>
<td>1.0</td>
<td>1.0</td>
<td>1.0</td>
<td>1.75</td>
<td>2.75</td>
</tr>

<tr>
<td>num_tool_calls</td>
<td>1.0</td>
<td>1.5275</td>
<td>6</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>1.5</td>
<td>3.5</td>
</tr>

<tr>
<td>num_errors</td>
<td>0.1667</td>
<td>0.3727</td>
<td>6</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.75</td>
</tr>

</table>


<h2>Examples <span class="muted">(showing up to 6 of 6)</span></h2>
<div class="examples">
<table>
<tr><th>#</th><th>reward</th><th>correct_answer_reward_func</th><th>completion</th></tr>

<tr>
<td>0</td>
<td>0.9</td>
<td>1.0</td>
<td><pre>The solution for the system of equations is:
\[
a = -3, \quad b = 3.
\]

Now, let&#39;s find the sum \( a + b \):
\[
a + b = -3 + 3 = 0.
\]

Therefore, \( a + b = 0 \).</pre></td>
</tr>

<tr>
<td>1</td>
<td>0.0</td>
<td>0.0</td>
<td><pre>Let&#39;s define the variables:
- \( m \) = number of band members in each row initially,
- \( r \) = number of rows initially,
- \( N \) = total number of band members in the band.

From the problem:
1. The band has less than 100 members, so \( N &lt; 100 \).
2. Initially, the band is arranged in a recta…</pre></td>
</tr>

<tr>
<td>2</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial given is:

\[
4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9
\]

To find the degree of the polynomial, we need to identify the term with the highest power of \(x\).

The terms are:
- \(4\) (constant term, degree 0)
- \(5x^3\) (degree 3)
- \(100\) (constant term, degree 0)
- \(2\pi x^4…</pre></td>
</tr>

<tr>
<td>3</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The solutions to the continuity equations are \(a = -3\) and \(b = 3\). Therefore, the sum \(a + b = -3 + 3 = 0\).

So, \(a + b = 0\).</pre></td>
</tr>

<tr>
<td>4</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>Let&#39;s denote:
- \( m \) = number of band members in each row in the original formation,
- \( r \) = number of rows in the original formation,
- \( N \) = total number of band members in the band.

From the problem:
1. The band has less than 100 members, so \( N &lt; 100 \).
2. When arranged in the ori…</pre></td>
</tr>

<tr>
<td>5</td>
<td>1.0</td>
<td>1.0</td>
<td><pre>The polynomial is \( 4 + 5x^3 + 100 + 2\pi x^4 + \sqrt{10} x^4 + 9 \).

To find the degree, we look for the highest power of \( x \) with a nonzero coefficient.

- \(4\), \(100\), and \(9\) are constants (degree 0).
- \(5x^3\) has degree 3.
- \(2\pi x^4\) has degree 4.
- \(\sqrt{10} x^4\) has degre…</pre></td>
</tr>

</table>
</div>
</details>
</details>
<!-- vf:end:reports -->
