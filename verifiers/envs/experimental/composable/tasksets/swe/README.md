# SWE Tasksets

## Legend

- Image sample: sample task images checked against `prime images list`.
  ✅ means every sampled image was found; ❌ means no sampled image was found.
- Validation: ✅ repeated no-op and gold-patch validation passed with
  [`SWEDebugEnv`](../../../../../../docs/environments.md#integrations-and-experimental-environments),
  — not yet complete.

## Progress

<table>
  <thead>
    <tr>
      <th>Backend</th>
      <th>Source</th>
      <th>Default HF dataset</th>
      <th>Original</th>
      <th>Filtered</th>
      <th>Image sample</th>
      <th>Validation</th>
      <th>Prime-data PRs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>swebench</code></td>
      <td><a href="https://arxiv.org/abs/2310.06770">paper</a></td>
      <td><a href="https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified"><code>princeton-nlp/SWE-bench_Verified</code></a></td>
      <td>500</td>
      <td>500</td>
      <td>❌ 0/3 found</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>r2e</code></td>
      <td><a href="https://arxiv.org/abs/2504.07164">paper</a></td>
      <td><a href="https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset"><code>R2E-Gym/R2E-Gym-Subset</code></a></td>
      <td>4,578</td>
      <td>4,578</td>
      <td>❌ 0/3 found</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>multiswe</code></td>
      <td><a href="https://arxiv.org/abs/2504.02605">paper</a></td>
      <td><a href="https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL"><code>PrimeIntellect/Multi-SWE-RL</code></a></td>
      <td>
        <strong>4,703</strong>
        <ul>
          <li><code>c</code>: 377</li>
          <li><code>cpp</code>: 449</li>
          <li><code>go</code>: 1,664</li>
          <li><code>java</code>: 976</li>
          <li><code>js</code>: 614</li>
          <li><code>rust</code>: 215</li>
          <li><code>ts</code>: 408</li>
        </ul>
      </td>
      <td>
        <strong>4,703</strong>
        <ul>
          <li><code>c</code>: 377</li>
          <li><code>cpp</code>: 449</li>
          <li><code>go</code>: 1,664</li>
          <li><code>java</code>: 976</li>
          <li><code>js</code>: 614</li>
          <li><code>rust</code>: 215</li>
          <li><code>ts</code>: 408</li>
        </ul>
      </td>
      <td>❌ 0/3 found</td>
      <td>—</td>
      <td><a href="https://github.com/PrimeIntellect-ai/prime-data/pull/6">#6</a></td>
    </tr>
    <tr>
      <td><code>openswe</code></td>
      <td><a href="https://arxiv.org/abs/2603.13023">paper</a></td>
      <td><a href="https://huggingface.co/datasets/GAIR/OpenSWE"><code>GAIR/OpenSWE</code></a> <code>openswe_oss</code></td>
      <td>45,320</td>
      <td>36,884</td>
      <td>✅ 4/4 found</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td><code>scaleswe</code></td>
      <td><a href="https://arxiv.org/abs/2602.09892">paper</a></td>
      <td><a href="https://huggingface.co/datasets/PrimeIntellect/Scale-SWE"><code>PrimeIntellect/Scale-SWE</code></a></td>
      <td>20,181</td>
      <td>17,202</td>
      <td>❌ 0/3 found</td>
      <td>✅</td>
      <td><a href="https://github.com/PrimeIntellect-ai/prime-data/pull/31">#31</a></td>
    </tr>
    <tr>
      <td><code>swelego-real</code></td>
      <td><a href="https://arxiv.org/abs/2601.01426">paper</a></td>
      <td><a href="https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data"><code>PrimeIntellect/SWE-Lego-Real-Data</code></a> <code>resolved</code></td>
      <td>5,009</td>
      <td>4,432</td>
      <td>❌ 0/3 found</td>
      <td>—</td>
      <td><a href="https://github.com/PrimeIntellect-ai/prime-data/pull/17">#17</a></td>
    </tr>
    <tr>
      <td><code>swerebench-v2</code></td>
      <td><a href="https://arxiv.org/abs/2602.23866">paper</a></td>
      <td><a href="https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Clean"><code>PrimeIntellect/SWE-rebench-V2-Clean</code></a></td>
      <td>
        <strong>32,079</strong>
        <ul>
          <li><code>c</code>: 230</li>
          <li><code>clojure</code>: 105</li>
          <li><code>cpp</code>: 182</li>
          <li><code>csharp</code>: 173</li>
          <li><code>dart</code>: 251</li>
          <li><code>elixir</code>: 416</li>
          <li><code>go</code>: 6,144</li>
          <li><code>java</code>: 1,716</li>
          <li><code>js</code>: 4,138</li>
          <li><code>julia</code>: 793</li>
          <li><code>kotlin</code>: 889</li>
          <li><code>lua</code>: 39</li>
          <li><code>ocaml</code>: 58</li>
          <li><code>php</code>: 1,445</li>
          <li><code>python</code>: 7,243</li>
          <li><code>r</code>: 157</li>
          <li><code>rust</code>: 3,123</li>
          <li><code>scala</code>: 411</li>
          <li><code>swift</code>: 362</li>
          <li><code>ts</code>: 4,204</li>
        </ul>
      </td>
      <td>
        <strong>6,304</strong>
        <ul>
          <li><code>c</code>: 13</li>
          <li><code>csharp</code>: 27</li>
          <li><code>dart</code>: 4</li>
          <li><code>elixir</code>: 84</li>
          <li><code>go</code>: 1,244</li>
          <li><code>java</code>: 324</li>
          <li><code>js</code>: 811</li>
          <li><code>kotlin</code>: 217</li>
          <li><code>lua</code>: 5</li>
          <li><code>ocaml</code>: 2</li>
          <li><code>php</code>: 237</li>
          <li><code>python</code>: 1,952</li>
          <li><code>r</code>: 51</li>
          <li><code>rust</code>: 477</li>
          <li><code>scala</code>: 58</li>
          <li><code>swift</code>: 64</li>
          <li><code>ts</code>: 734</li>
        </ul>
      </td>
      <td>❌ 0/3 found</td>
      <td>✅</td>
      <td>
        <a href="https://github.com/PrimeIntellect-ai/prime-data/pull/20">#20</a>,
        <a href="https://github.com/PrimeIntellect-ai/prime-data/pull/23">#23</a>
      </td>
    </tr>
    <tr>
      <td><code>swesmith-*</code></td>
      <td><a href="https://arxiv.org/abs/2504.21798">paper</a></td>
      <td><a href="https://huggingface.co/datasets/SWE-bench/SWE-smith-py"><code>SWE-bench/SWE-smith-*</code></a></td>
      <td>
        <strong>88,130</strong>
        <ul>
          <li><code>py</code>: 50,908</li>
          <li><code>go</code>: 8,212</li>
          <li><code>java</code>: 7,470</li>
          <li><code>js</code>: 6,073</li>
          <li><code>ts</code>: 5,032</li>
          <li><code>rs</code>: 5,311</li>
          <li><code>cpp</code>: 5,123</li>
          <li><code>php</code>: 1</li>
        </ul>
      </td>
      <td>
        <strong>83,519</strong>
        <ul>
          <li><code>py</code>: 50,908</li>
          <li><code>go</code>: 8,212</li>
          <li><code>java</code>: 7,470</li>
          <li><code>js</code>: 6,073</li>
          <li><code>ts</code>: 5,032</li>
          <li><code>rs</code>: 5,311</li>
          <li><code>cpp</code>: 512</li>
          <li><code>php</code>: 1</li>
        </ul>
      </td>
      <td>❌ 0/8 found</td>
      <td>—</td>
      <td>—</td>
    </tr>
  </tbody>
</table>

## Workflow

1. Add or port the taskset under this directory and register its backend in
   [`make_swe_taskset(...)`](swe_tasksets.py).
2. Prefer the upstream dataset shape and evaluation lifecycle, then publish a
   filtered Prime dataset through `prime-data` when validation identifies rows
   to exclude.
3. Mirror task images that will run at scale into the Prime image registry so
   sandbox startup uses quick pulls and large sweeps avoid upstream registry
   rate limits.
4. Validate with
   [`SWEDebugEnv`](../../swe_debug_env.py): no-op runs should fail real tasks,
   gold-patch runs should pass, and repeated passes should separate task
   quality issues from sandbox or infrastructure failures.
