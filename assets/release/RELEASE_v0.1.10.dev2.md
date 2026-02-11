# Verifiers v0.1.10.dev1 Release Notes

*Date:* 02/04/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.10.dev0...v0.1.10.dev1

## Changes since v0.1.10.dev0

- info oaitools fix (#821)
- Capture stdout/stderr for live display (#819)
- track token usage in eval (#816)
- RLM: show full user message (#818)
- remove filesystem info from rlm system prompts (#817)
- RLM: add prompt verbosity parameters (#814)
- re-raise auth errs + fix overlong prompt err for openrouter (#813)
- add default sandbox_labels to rlm-secrets (#810)
- Improve vf-eval display (#809)
- Increase Sandbox Default Thread Worker Count (#807)
- Tool content validation (#806)
- env server (#799)
- Add Browser Env Integration (#732)
- adjust CliAgentEnv sandbox creation timeout + remove DummyHarborEnv (#804)
- verifiers: fix tool call rendering from saved outputs (#802)
- RLM: remove code jail -> simplify code (#800)
- Add sync bulk sandbox teardown for RLM env (#798)
- overhaul saving outputs (#774)
- Propagate RLM stop errors from root and sub tools (#797)
- clean up on task cancelation to avoid resource leakage (#795)
- `CliAgentEnv`: teardown sandboxes via bulk delete (#796)
- RLM: Fix trajectory collision (#786)
- lazy import datasets (#794)
- add rLLM integration to docs
- cancel outstanding tasks if one task raises in `generate` (#793)
- revert wiki-search
- update environments/README.md (#790)
- util for enforcing env vars are set (#789)
- return last result if retries exhausted (#782)
- warning log in RLTrainer (#783)
- RLM: Simplify code (#781)
- hello world tasks for TerminusHarborEnv and OpenCodeHarborEnv (#775)
- fix retry for invalid model response errors (#778)
- Make local RLM REPL concurrency configurable (#777)
- RLM: re-enable Sandboxes for both Python and Bash (#776)
- fix alphabet-sort
- fix alphabet-sort
- raise on empty response error (openrouter) to trigger retries (#772)
- mirror cli in toml config (#773)
- fix double save results in vf-eval (#771)
- remove log file for cli agent env (#770)
- eval --debug mode to skip Rich (#769)
- tools eval example
- Harbor examples (#766)
- Feature: Add tools metadata for eval viewer (#767)
- expose sandbox labels in `SandboxEnv` and `CliAgentEnv` (#768)
- RLM env stop condition fix (#757)
- lazy init locked chromadb instance in wiki-search (#765)
- created the rlm_secrets environment (#763)
- Move RLM system prompt into first user prompt (#764)
- gepa dep
- integrated gepa training, ui to track (#747)
- prime tunnel in cliagentenv (#746)
- RLM: make bash REPL default, keep Python REPL optional (#758)
- Sebastian/rlm file system 2026 01 20 (#756)
- eval tui (#735)
- multi-env evals config (#734)
- Add retry support for infrastructure errors in vf-eval (#750)
- RLM: tools, sub_tools, root_tools (#749)
- optional DatasetBuilder pattern (#739)
- Add copy mode to vf-tui (#745)
- RLMEnv: Make sub-LLM calls work for training (#738)
