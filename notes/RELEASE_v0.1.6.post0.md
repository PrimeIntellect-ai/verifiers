# Verifiers v0.1.6.post0 Release Notes

*Date:* 10/20/25

**Post-Release Update**: Quick bug fix for multi-turn chat template handling.

Verifiers v0.1.6 primarly focuses on a refactor of the evaluation logic (`vf-eval`) and generation methods (in `vf.Environment`) to track more metadata, streamline duplicated logic, enable intermediate saving of generations, and allow for more flexible evaluation workflows (i.e. importable utilities in `verifiers.utils.eval_utils`). 

The main **breaking change** is that `vf.Environment.generate` and `vf.Environment.evaluate` are now async methods, with `generate_sync` and `evaluate_sync` included as synchronous wrappers. 

We are also migrating towards using the `state` object more explicitly to track information throughout rollouts; existing workflows should be unaffected, but we encourage users to migrate to the new `state` object for better tracking of information throughout rollouts, as eventually other arguments will be deprecated.


**Full Changelog**: https://github.com/willccbb/verifiers/compare/v0.1.6...v0.1.6.post0