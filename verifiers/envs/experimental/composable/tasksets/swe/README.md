# SWE Tasksets

## Legend

- Prime images: ✅ means task images are available in our registry; ❌ means
  they are not known to be available there yet.
- Validation: ✅ means the taskset has a linked `prime-data` PR and was
  validated with
  [`SandboxDebugEnv`](../../../../../../docs/environments.md#integrations-and-experimental-environments),
  — not yet complete.

## Progress

<table>
  <thead>
    <tr>
      <th>Backend</th>
      <th>Source</th>
      <th>Default HF dataset</th>
      <th>Language</th>
      <th>Original</th>
      <th>Filtered</th>
      <th>Prime images</th>
      <th>Validation</th>
      <th>Prime-data PRs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>swebench</code></td>
      <td><a href="https://arxiv.org/abs/2310.06770">paper</a></td>
      <td><a href="https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified"><code>princeton-nlp/SWE-bench_Verified</code></a></td>
      <td><code>python</code></td>
      <td>500</td>
      <td>500</td>
      <td>✅</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>r2e</code></td>
      <td><a href="https://arxiv.org/abs/2504.07164">paper</a></td>
      <td><a href="https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset"><code>R2E-Gym/R2E-Gym-Subset</code></a></td>
      <td><code>python</code></td>
      <td>4,578</td>
      <td>4,578</td>
      <td>✅</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td rowspan="8"><code>multiswe</code></td>
      <td rowspan="8"><a href="https://arxiv.org/abs/2504.02605">paper</a></td>
      <td rowspan="8"><a href="https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL"><code>PrimeIntellect/Multi-SWE-RL</code></a></td>
      <td><strong>all</strong></td>
      <td><strong>4,703</strong></td>
      <td><strong>4,703</strong></td>
      <td rowspan="8">✅</td>
      <td rowspan="8">✅</td>
      <td rowspan="8"><a href="https://github.com/PrimeIntellect-ai/prime-data/pull/6">#6</a></td>
    </tr>
    <tr>
      <td><code>c</code></td>
      <td>377</td>
      <td>377</td>
    </tr>
    <tr>
      <td><code>cpp</code></td>
      <td>449</td>
      <td>449</td>
    </tr>
    <tr>
      <td><code>go</code></td>
      <td>1,664</td>
      <td>1,664</td>
    </tr>
    <tr>
      <td><code>java</code></td>
      <td>976</td>
      <td>976</td>
    </tr>
    <tr>
      <td><code>js</code></td>
      <td>614</td>
      <td>614</td>
    </tr>
    <tr>
      <td><code>rust</code></td>
      <td>215</td>
      <td>215</td>
    </tr>
    <tr>
      <td><code>ts</code></td>
      <td>408</td>
      <td>408</td>
    </tr>
    <tr>
      <td><code>openswe</code></td>
      <td><a href="https://arxiv.org/abs/2603.13023">paper</a></td>
      <td><a href="https://huggingface.co/datasets/GAIR/OpenSWE"><code>GAIR/OpenSWE</code></a> <code>openswe_oss</code></td>
      <td><code>python</code></td>
      <td>45,320</td>
      <td>36,884</td>
      <td>❌</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>scaleswe</code></td>
      <td><a href="https://arxiv.org/abs/2602.09892">paper</a></td>
      <td><a href="https://huggingface.co/datasets/PrimeIntellect/Scale-SWE"><code>PrimeIntellect/Scale-SWE</code></a></td>
      <td><code>python</code></td>
      <td>20,181</td>
      <td>17,202</td>
      <td>✅</td>
      <td>✅</td>
      <td><a href="https://github.com/PrimeIntellect-ai/prime-data/pull/31">#31</a></td>
    </tr>
    <tr>
      <td><code>swelego-real</code></td>
      <td><a href="https://arxiv.org/abs/2601.01426">paper</a></td>
      <td><a href="https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data"><code>PrimeIntellect/SWE-Lego-Real-Data</code></a> <code>resolved</code></td>
      <td><code>python</code></td>
      <td>5,009</td>
      <td>4,432</td>
      <td>✅</td>
      <td>✅</td>
      <td><a href="https://github.com/PrimeIntellect-ai/prime-data/pull/17">#17</a></td>
    </tr>
    <tr>
      <td rowspan="21"><code>swerebench-v2</code></td>
      <td rowspan="21"><a href="https://arxiv.org/abs/2602.23866">paper</a></td>
      <td rowspan="21"><a href="https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Clean"><code>PrimeIntellect/SWE-rebench-V2-Clean</code></a></td>
      <td><strong>all</strong></td>
      <td><strong>32,079</strong></td>
      <td><strong>6,304</strong></td>
      <td rowspan="21">✅</td>
      <td rowspan="21">✅</td>
      <td rowspan="21">
        <a href="https://github.com/PrimeIntellect-ai/prime-data/pull/20">#20</a>,
        <a href="https://github.com/PrimeIntellect-ai/prime-data/pull/23">#23</a>
      </td>
    </tr>
    <tr>
      <td><code>c</code></td>
      <td>230</td>
      <td>13</td>
    </tr>
    <tr>
      <td><code>clojure</code></td>
      <td>105</td>
      <td>0</td>
    </tr>
    <tr>
      <td><code>cpp</code></td>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <td><code>csharp</code></td>
      <td>173</td>
      <td>27</td>
    </tr>
    <tr>
      <td><code>dart</code></td>
      <td>251</td>
      <td>4</td>
    </tr>
    <tr>
      <td><code>elixir</code></td>
      <td>416</td>
      <td>84</td>
    </tr>
    <tr>
      <td><code>go</code></td>
      <td>6,144</td>
      <td>1,244</td>
    </tr>
    <tr>
      <td><code>java</code></td>
      <td>1,716</td>
      <td>324</td>
    </tr>
    <tr>
      <td><code>js</code></td>
      <td>4,138</td>
      <td>811</td>
    </tr>
    <tr>
      <td><code>julia</code></td>
      <td>793</td>
      <td>0</td>
    </tr>
    <tr>
      <td><code>kotlin</code></td>
      <td>889</td>
      <td>217</td>
    </tr>
    <tr>
      <td><code>lua</code></td>
      <td>39</td>
      <td>5</td>
    </tr>
    <tr>
      <td><code>ocaml</code></td>
      <td>58</td>
      <td>2</td>
    </tr>
    <tr>
      <td><code>php</code></td>
      <td>1,445</td>
      <td>237</td>
    </tr>
    <tr>
      <td><code>python</code></td>
      <td>7,243</td>
      <td>1,952</td>
    </tr>
    <tr>
      <td><code>r</code></td>
      <td>157</td>
      <td>51</td>
    </tr>
    <tr>
      <td><code>rust</code></td>
      <td>3,123</td>
      <td>477</td>
    </tr>
    <tr>
      <td><code>scala</code></td>
      <td>411</td>
      <td>58</td>
    </tr>
    <tr>
      <td><code>swift</code></td>
      <td>362</td>
      <td>64</td>
    </tr>
    <tr>
      <td><code>ts</code></td>
      <td>4,204</td>
      <td>734</td>
    </tr>
    <tr>
      <td rowspan="9"><code>swesmith-*</code></td>
      <td rowspan="9"><a href="https://arxiv.org/abs/2504.21798">paper</a></td>
      <td rowspan="9"><a href="https://huggingface.co/datasets/SWE-bench/SWE-smith-py"><code>SWE-bench/SWE-smith-*</code></a></td>
      <td><strong>all</strong></td>
      <td><strong>88,130</strong></td>
      <td><strong>83,519</strong></td>
      <td rowspan="9">❌</td>
      <td rowspan="9">—</td>
      <td rowspan="9">—</td>
    </tr>
    <tr>
      <td><code>py</code></td>
      <td>50,908</td>
      <td>50,908</td>
    </tr>
    <tr>
      <td><code>go</code></td>
      <td>8,212</td>
      <td>8,212</td>
    </tr>
    <tr>
      <td><code>java</code></td>
      <td>7,470</td>
      <td>7,470</td>
    </tr>
    <tr>
      <td><code>js</code></td>
      <td>6,073</td>
      <td>6,073</td>
    </tr>
    <tr>
      <td><code>ts</code></td>
      <td>5,032</td>
      <td>5,032</td>
    </tr>
    <tr>
      <td><code>rs</code></td>
      <td>5,311</td>
      <td>5,311</td>
    </tr>
    <tr>
      <td><code>cpp</code></td>
      <td>5,123</td>
      <td>512</td>
    </tr>
    <tr>
      <td><code>php</code></td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

## Workflow

1. Add or port the taskset under this directory and register its backend in
   [`make_swe_taskset(...)`](swe_tasksets.py).
2. Mirror task images that will run at scale into the Prime image registry so
   sandbox startup uses quick pulls and large sweeps avoid upstream registry
   rate limits.
3. Prefer the upstream dataset shape and evaluation lifecycle, then publish a
   filtered Prime dataset through `prime-data` when validation identifies rows
   to exclude.
4. Validate with
   [`SandboxDebugEnv`](../../sandbox_debug_env.py): no-op runs should fail real tasks,
   gold-patch runs should pass, and repeated passes should separate task
   quality issues from sandbox or infrastructure failures.
