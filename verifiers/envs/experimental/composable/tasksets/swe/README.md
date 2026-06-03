# SWE Tasksets

## Legend

- Images in Prime registry: ✅ uploaded, 📝 image names collected for upload,
  — not tracked.
- Validation: ✅ repeated no-op and gold-patch validation passed with
  [`SWEDebugEnv`](../../../../../../docs/environments.md#integrations-and-experimental-environments),
  — not yet complete.

## Progress

| Backend | Source | Default HF dataset | Original | Filtered | Image status | Validation | Prime-data PRs |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `swebench` | [paper](https://arxiv.org/abs/2310.06770) | [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 500 | 500 | ✅ 500 uploaded | — | — |
| `r2e` | [paper](https://arxiv.org/abs/2504.07164) | [`R2E-Gym/R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | 4,578 | 4,578 | ✅ 4,578 uploaded | — | — |
| `multiswe` | [paper](https://arxiv.org/abs/2504.02605) | [`PrimeIntellect/Multi-SWE-RL`](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL) | <details><summary>4,703</summary><ul><li><code>c</code>: 377</li><li><code>cpp</code>: 449</li><li><code>go</code>: 1,664</li><li><code>java</code>: 976</li><li><code>js</code>: 614</li><li><code>rust</code>: 215</li><li><code>ts</code>: 408</li></ul></details> | <details><summary>4,703</summary><ul><li><code>c</code>: 377</li><li><code>cpp</code>: 449</li><li><code>go</code>: 1,664</li><li><code>java</code>: 976</li><li><code>js</code>: 614</li><li><code>rust</code>: 215</li><li><code>ts</code>: 408</li></ul></details> | 📝 4,818 listed for upload | — | [#6](https://github.com/PrimeIntellect-ai/prime-data/pull/6) |
| `openswe` | [paper](https://arxiv.org/abs/2603.13023) | [`GAIR/OpenSWE`](https://huggingface.co/datasets/GAIR/OpenSWE) `openswe_oss` | 45,320 | 36,884 | — | — | — |
| `scaleswe` | [paper](https://arxiv.org/abs/2602.09892) | [`PrimeIntellect/Scale-SWE`](https://huggingface.co/datasets/PrimeIntellect/Scale-SWE) | 20,181 | 17,202 | 📝 20,181 listed for upload | ✅ | [#31](https://github.com/PrimeIntellect-ai/prime-data/pull/31) |
| `swelego-real` | [paper](https://arxiv.org/abs/2601.01426) | [`PrimeIntellect/SWE-Lego-Real-Data`](https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data) `resolved` | 5,009 | 4,432 | — | — | [#17](https://github.com/PrimeIntellect-ai/prime-data/pull/17) |
| `swerebench-v2` | [paper](https://arxiv.org/abs/2602.23866) | [`PrimeIntellect/SWE-rebench-V2-Clean`](https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Clean) | <details><summary>32,079</summary><ul><li><code>c</code>: 230</li><li><code>clojure</code>: 105</li><li><code>cpp</code>: 182</li><li><code>csharp</code>: 173</li><li><code>dart</code>: 251</li><li><code>elixir</code>: 416</li><li><code>go</code>: 6,144</li><li><code>java</code>: 1,716</li><li><code>js</code>: 4,138</li><li><code>julia</code>: 793</li><li><code>kotlin</code>: 889</li><li><code>lua</code>: 39</li><li><code>ocaml</code>: 58</li><li><code>php</code>: 1,445</li><li><code>python</code>: 7,243</li><li><code>r</code>: 157</li><li><code>rust</code>: 3,123</li><li><code>scala</code>: 411</li><li><code>swift</code>: 362</li><li><code>ts</code>: 4,204</li></ul></details> | <details><summary>6,304</summary><ul><li><code>c</code>: 13</li><li><code>csharp</code>: 27</li><li><code>dart</code>: 4</li><li><code>elixir</code>: 84</li><li><code>go</code>: 1,244</li><li><code>java</code>: 324</li><li><code>js</code>: 811</li><li><code>kotlin</code>: 217</li><li><code>lua</code>: 5</li><li><code>ocaml</code>: 2</li><li><code>php</code>: 237</li><li><code>python</code>: 1,952</li><li><code>r</code>: 51</li><li><code>rust</code>: 477</li><li><code>scala</code>: 58</li><li><code>swift</code>: 64</li><li><code>ts</code>: 734</li></ul></details> | ✅ 6,304 uploaded | ✅ | [#20](https://github.com/PrimeIntellect-ai/prime-data/pull/20), [#23](https://github.com/PrimeIntellect-ai/prime-data/pull/23) |
| `swesmith-*` | [paper](https://arxiv.org/abs/2504.21798) | [`SWE-bench/SWE-smith-*`](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) | <details><summary>88,130</summary><ul><li><code>py</code>: 50,908</li><li><code>go</code>: 8,212</li><li><code>java</code>: 7,470</li><li><code>js</code>: 6,073</li><li><code>ts</code>: 5,032</li><li><code>rs</code>: 5,311</li><li><code>cpp</code>: 5,123</li><li><code>php</code>: 1</li></ul></details> | <details><summary>83,519</summary><ul><li><code>py</code>: 50,908</li><li><code>go</code>: 8,212</li><li><code>java</code>: 7,470</li><li><code>js</code>: 6,073</li><li><code>ts</code>: 5,032</li><li><code>rs</code>: 5,311</li><li><code>cpp</code>: 512</li><li><code>php</code>: 1</li></ul></details> | — | — | — |

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
