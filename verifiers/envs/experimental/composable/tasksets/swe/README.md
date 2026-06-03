# SWE Tasksets

Progress snapshot for the default `make_swe_taskset(...)` backends as of
2026-06-03.

Image status: ✅ mirrored, 📋 transfer list, — none.

| Backend | Source | Default HF dataset | Original | Filtered | Prime images | Multi-valid | Prime-data PRs |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `swebench` | [paper](https://arxiv.org/abs/2310.06770) | [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 500 | 500 | ✅ 500 | — | — |
| `r2e` | [paper](https://arxiv.org/abs/2504.07164) | [`R2E-Gym/R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | 4,578 | 4,578 | ✅ 4,578 | — | — |
| `multiswe` | [paper](https://arxiv.org/abs/2504.02605) | [`PrimeIntellect/Multi-SWE-RL`](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL) | 4,703 | 4,703<br>`c` 377, `cpp` 449, `go` 1,664, `java` 976, `js` 614, `rust` 215, `ts` 408 | 📋 4,818 | — | [#6](https://github.com/PrimeIntellect-ai/prime-data/pull/6) |
| `openswe` | [paper](https://arxiv.org/abs/2603.13023) | [`GAIR/OpenSWE`](https://huggingface.co/datasets/GAIR/OpenSWE) `openswe_oss` | 45,320 | 36,884 | — | — | — |
| `scaleswe` | [paper](https://arxiv.org/abs/2602.09892) | [`PrimeIntellect/Scale-SWE`](https://huggingface.co/datasets/PrimeIntellect/Scale-SWE) | 20,181 | 17,202 | 📋 20,181 | ✅ | [#31](https://github.com/PrimeIntellect-ai/prime-data/pull/31) |
| `swelego-real` | [paper](https://arxiv.org/abs/2601.01426) | [`PrimeIntellect/SWE-Lego-Real-Data`](https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data) `resolved` | 5,009 | 4,432 | — | — | [#17](https://github.com/PrimeIntellect-ai/prime-data/pull/17) |
| `swerebench-v2` | [paper](https://arxiv.org/abs/2602.23866) | [`PrimeIntellect/SWE-rebench-V2-Clean`](https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Clean) | 32,079 | 6,304<br>`c` 13, `csharp` 27, `dart` 4, `elixir` 84, `go` 1,244, `java` 324, `js` 811, `kotlin` 217, `lua` 5, `ocaml` 2, `php` 237, `python` 1,952, `r` 51, `rust` 477, `scala` 58, `swift` 64, `ts` 734 | ✅ 6,304 | ✅ | [#20](https://github.com/PrimeIntellect-ai/prime-data/pull/20), [#23](https://github.com/PrimeIntellect-ai/prime-data/pull/23) |
| `swesmith-*` | [paper](https://arxiv.org/abs/2504.21798) | [`SWE-bench/SWE-smith-*`](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) | 88,130<br>`py` 50,908, `go` 8,212, `java` 7,470, `js` 6,073, `ts` 5,032, `rs` 5,311, `cpp` 5,123, `php` 1 | 83,519<br>`py` 50,908, `go` 8,212, `java` 7,470, `js` 6,073, `ts` 5,032, `rs` 5,311, `cpp` 512, `php` 1 | — | — | — |
