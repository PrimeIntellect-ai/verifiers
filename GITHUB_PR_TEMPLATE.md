## Description
Add sparse metrics support for mathematically correct domain averaging in multi-domain environments. This feature enables selective averaging that excludes irrelevant zero values, solving the domain dilution problem in composite evaluation environments like ProfBench.

**Key improvements:**
- Chemistry domain: `avg - 72.9 (relevant: 2/12)` instead of diluted `avg - 12.3`
- Physics domain: `avg - 66.2 (relevant: 10/12)` instead of diluted `avg - 56.2`
- Visual distinction: Shows `-` for sparse values instead of misleading `0.0`

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Test improvement

## Testing
- [ ] All existing tests pass when running `uv run pytest` locally.
- [ ] New tests have been added to cover the changes

**Manual Testing:**
Tested with ProfBench environment showing correct sparse metrics behavior:
- Domain-specific averages exclude irrelevant metrics
- Sparse values display as `-` in output
- `(relevant: X/Y)` info shows sparsity clearly

## Checklist
- [ ] My code follows the style guidelines of this project as outlined in [AGENTS.md](https://github.com/PrimeIntellect-ai/verifiers/blob/main/AGENTS.md)
- [x] I have performed a self-review of my own code
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Additional Notes

### Implementation Overview
- **Types**: Added `sparse_metrics` fields to `RolloutScore`, `RolloutScores`, and `GenerateOutputs`
- **Environment**: Sparse tracking during interleaved scoring in `generate()` method
- **Rubrics**: Batch scoring with missing metrics marked as sparse in `score_rollouts()`
- **EnvGroup**: New `EnvGroupSparseRubric` class with `enable_sparse_metrics=True` opt-in
- **Display**: Sparse-aware averaging and `-` display in `eval_utils.py`

### Backwards Compatibility
✅ Zero breaking changes - all existing environments work unchanged  
✅ Opt-in only - sparse metrics activate only with `enable_sparse_metrics=True`  
✅ Default behavior preserved - standard averaging remains identical  

### Testing Instructions
To test with ProfBench:
```bash
# 1. Clone ProfBench with sparse support
#  ( this is a env. bounty that is in progrss of being ipmlemented, which needed this PR )
git clone https://github.com/vxnuaj/prime-environments.git -b vxnuaj/profbench
cd prime-environments

# 2. Clone this verifiers fork / pr branch
git clone https://github.com/vxnuaj/verifiers.git -b vxnuaj/dynamic-sparse-rewards
cd verifiers

# 3. Install and test
cd ..
uv pip install -e .
vf-eval -s profbench -m gpt-4.1-mini --env-args '{"judge_model": "openai/gpt-4.1-mini"}' -n 12 -r 1
```

**Expected output:** Domain averages like `chemistry_phd_reward: avg - 72.9 (relevant: 2/12)` with `-` showing sparse values.
