# SWE Tasksets

## Legend

- Prime images: ✅ mirrored, 📋 transfer list, — not tracked.
- Validation: ✅ repeated no-op and gold-patch validation passed with
  [`SWEDebugEnv`](../../../../../../docs/environments.md#integrations-and-experimental-environments),
  — not yet complete.

## Progress

| Backend | Source | Default HF dataset | Original | Filtered | Prime images | Validation | Prime-data PRs |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `swebench` | [paper](https://arxiv.org/abs/2310.06770) | [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 500 | 500 | ✅ 500 | — | — |
| `r2e` | [paper](https://arxiv.org/abs/2504.07164) | [`R2E-Gym/R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | 4,578 | 4,578 | ✅ 4,578 | — | — |
| `multiswe` | [paper](https://arxiv.org/abs/2504.02605) | [`PrimeIntellect/Multi-SWE-RL`](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL) | 4,703 | 4,703 | 📋 4,818 | — | [#6](https://github.com/PrimeIntellect-ai/prime-data/pull/6) |
| `openswe` | [paper](https://arxiv.org/abs/2603.13023) | [`GAIR/OpenSWE`](https://huggingface.co/datasets/GAIR/OpenSWE) `openswe_oss` | 45,320 | 36,884 | — | — | — |
| `scaleswe` | [paper](https://arxiv.org/abs/2602.09892) | [`PrimeIntellect/Scale-SWE`](https://huggingface.co/datasets/PrimeIntellect/Scale-SWE) | 20,181 | 17,202 | 📋 20,181 | ✅ | [#31](https://github.com/PrimeIntellect-ai/prime-data/pull/31) |
| `swelego-real` | [paper](https://arxiv.org/abs/2601.01426) | [`PrimeIntellect/SWE-Lego-Real-Data`](https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data) `resolved` | 5,009 | 4,432 | — | — | [#17](https://github.com/PrimeIntellect-ai/prime-data/pull/17) |
| `swerebench-v2` | [paper](https://arxiv.org/abs/2602.23866) | [`PrimeIntellect/SWE-rebench-V2-Clean`](https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Clean) | 32,079 | 6,304 | ✅ 6,304 | ✅ | [#20](https://github.com/PrimeIntellect-ai/prime-data/pull/20), [#23](https://github.com/PrimeIntellect-ai/prime-data/pull/23) |
| `swesmith-*` | [paper](https://arxiv.org/abs/2504.21798) | [`SWE-bench/SWE-smith-*`](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) | 88,130 | 83,519 | — | — | — |

## Language Breakdown

| Backend | Language | Original | Filtered |
| --- | --- | ---: | ---: |
| `multiswe` | `c` | 377 | 377 |
| `multiswe` | `cpp` | 449 | 449 |
| `multiswe` | `go` | 1,664 | 1,664 |
| `multiswe` | `java` | 976 | 976 |
| `multiswe` | `js` | 614 | 614 |
| `multiswe` | `rust` | 215 | 215 |
| `multiswe` | `ts` | 408 | 408 |
| `swerebench-v2` | `c` | 230 | 13 |
| `swerebench-v2` | `clojure` | 105 | 0 |
| `swerebench-v2` | `cpp` | 182 | 0 |
| `swerebench-v2` | `csharp` | 173 | 27 |
| `swerebench-v2` | `dart` | 251 | 4 |
| `swerebench-v2` | `elixir` | 416 | 84 |
| `swerebench-v2` | `go` | 6,144 | 1,244 |
| `swerebench-v2` | `java` | 1,716 | 324 |
| `swerebench-v2` | `js` | 4,138 | 811 |
| `swerebench-v2` | `julia` | 793 | 0 |
| `swerebench-v2` | `kotlin` | 889 | 217 |
| `swerebench-v2` | `lua` | 39 | 5 |
| `swerebench-v2` | `ocaml` | 58 | 2 |
| `swerebench-v2` | `php` | 1,445 | 237 |
| `swerebench-v2` | `python` | 7,243 | 1,952 |
| `swerebench-v2` | `r` | 157 | 51 |
| `swerebench-v2` | `rust` | 3,123 | 477 |
| `swerebench-v2` | `scala` | 411 | 58 |
| `swerebench-v2` | `swift` | 362 | 64 |
| `swerebench-v2` | `ts` | 4,204 | 734 |
| `swesmith-*` | `py` | 50,908 | 50,908 |
| `swesmith-*` | `go` | 8,212 | 8,212 |
| `swesmith-*` | `java` | 7,470 | 7,470 |
| `swesmith-*` | `js` | 6,073 | 6,073 |
| `swesmith-*` | `ts` | 5,032 | 5,032 |
| `swesmith-*` | `rs` | 5,311 | 5,311 |
| `swesmith-*` | `cpp` | 5,123 | 512 |
| `swesmith-*` | `php` | 1 | 1 |

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
