# SWE Tasksets

This directory contains the SWE-style `SandboxTaskSet` backends available via
`make_swe_taskset(...)`. Counts below are for the default dataset/config/split
used by each taskset as of 2026-06-03. User-provided `filter_fn` values can
further reduce the effective rollout set.

The image column records whether the default taskset points at images mirrored
under `us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox`, or at
another upstream registry. Where a Prime image transfer list exists, the list
name is called out explicitly.

| Backend | Default HF dataset | Original count | Filtered count | Images uploaded in Prime registry / list | Multiple validations passed | Prime-data PRs |
| --- | --- | ---: | ---: | --- | --- | --- |
| `swebench` | [`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 500 | 500 | 500 task images, derived from SWE-bench `instance_image_key` and loaded from `prod-sandbox/swebench/...`; no separate prime-data image list. | No Prime multi-pass validation recorded; uses upstream SWE-bench Verified curation. | N/A |
| `r2e` | [`R2E-Gym/R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) | 4,578 | 4,578 | 4,578 unique `docker_image` values loaded from `prod-sandbox/...`; no separate prime-data image list. | No Prime multi-pass validation recorded. | N/A |
| `multiswe` | [`PrimeIntellect/Multi-SWE-RL`](https://huggingface.co/datasets/PrimeIntellect/Multi-SWE-RL) | 4,703 normalized rows | 4,703 | 4,818 entries in `multi-swe-bench/scripts/images_rl.txt` for the mirrored `mswebench/...` image set. | No Prime multi-pass validation recorded for the default dataset. | [#6](https://github.com/PrimeIntellect-ai/prime-data/pull/6) |
| `openswe` | [`GAIR/OpenSWE`](https://huggingface.co/datasets/GAIR/OpenSWE), config `openswe_oss` | 45,320 total OpenSWE environments | 36,884 `openswe_oss` rows | Not Prime-mirrored in this taskset; rows use upstream `image_name`. | No Prime multi-pass validation recorded; upstream applies its own quality filtering. | N/A |
| `scaleswe` | [`PrimeIntellect/Scale-SWE`](https://huggingface.co/datasets/PrimeIntellect/Scale-SWE) | 20,181 | 17,202 | `scale_swe_images.txt` listed 20,181 upstream `aweaiteam/scaleswe:*` images for transfer; the current taskset still uses row `image_url`. The filtered HF dataset also ships `scale-swe-exclude-images.json`. | Yes: no-op validation, gold-patch validation, and lower-concurrency infra reruns; failing/flaky rows are filtered in the Prime fork. | [#31](https://github.com/PrimeIntellect-ai/prime-data/pull/31) |
| `swelego-real` | [`PrimeIntellect/SWE-Lego-Real-Data`](https://huggingface.co/datasets/PrimeIntellect/SWE-Lego-Real-Data), split `resolved` | 5,009 resolved rows | 4,432 | Not Prime-mirrored in this taskset; rows use public `jierun/sweb.eval.x86_64.*` images. | No Prime multi-pass validation recorded; filter removes truncated pytest IDs. | [#17](https://github.com/PrimeIntellect-ai/prime-data/pull/17) |
| `swerebench-v2` | [`PrimeIntellect/SWE-rebench-V2-Clean`](https://huggingface.co/datasets/PrimeIntellect/SWE-rebench-V2-Clean) | 32,079 | 6,304 | 6,304 filtered rows reference `prod-sandbox/swerebenchv2/...` after the Prime data script rewrites Docker Hub image names to the GCP mirror. | Yes: LLM metadata filter, image/lang blocklists, gold-patch validation, independent flaky revalidation, and no-edit pass exclusions. | [#20](https://github.com/PrimeIntellect-ai/prime-data/pull/20), [#23](https://github.com/PrimeIntellect-ai/prime-data/pull/23) |
| `swesmith-*` | [`SWE-bench/SWE-smith-{language}`](https://huggingface.co/SWE-bench) for `py`, `go`, `java`, `js`, `ts`, `rs`, `cpp`, `php` | 88,130 across default languages | 83,519 after C++ profile filtering | Not Prime-mirrored in this taskset; rows use upstream SWE-Smith `image_name`. | No Prime multi-pass validation recorded; taskset filters C++ rows without an upstream `RepoProfile`. | N/A |

## Notes

- `filtered count` is the dataset count used by the default Verifiers taskset,
  before any caller-supplied `filter_fn`.
- `scaleswe` validation artifacts live in the private HF dataset as
  `scale-swe-validation.jsonl` and `scale-swe-exclude-images.json`.
- `swerebench-v2` validation artifacts are maintained in `prime-data` next to
  `datasets/swe-rebench-v2-clean.py`; the published dataset card embeds that
  generation source.
- `swesmith-*` aggregate counts by language:
  `py` 50,908 -> 50,908, `go` 8,212 -> 8,212,
  `java` 7,470 -> 7,470, `js` 6,073 -> 6,073,
  `ts` 5,032 -> 5,032, `rs` 5,311 -> 5,311,
  `cpp` 5,123 -> 512, `php` 1 -> 1.
