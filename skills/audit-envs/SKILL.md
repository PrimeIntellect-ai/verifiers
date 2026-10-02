---
name: audit-envs
description: Check whether a Verifiers environment measures the intended ability. Use to review task instructions, setup, graders, model judges, answer leaks, reward shortcuts, or suspicious evaluation results.
---

# Audit Environments

Check that tasks ask for the right work, provide a usable setup, and give the
right scores. Start with source files and saved runs. A passing evaluation alone
does not show that the benchmark is sound.

## Fix the scope

Record the package and dataset revisions, selected tasks, images, harness,
runtime, and scoring settings. For saved runs, read `configs/resolved/eval.json`
and match the traces to the task version. Treat mismatched versions as a gap in
the evidence.

Count the tasks and episodes available, those inspected, and those with enough
evidence to judge. State whether the review covers the full set or a sample.
Do not report keyword matches as confirmed defects.

## Follow the task from prompt to score

Read `Taskset.load`, `TaskData`, task hooks, rewards, metrics, and configured
judges. Include custom `Env` scoring, tool servers, Dockerfiles, setup scripts,
and Harbor tests where used. Trace the actual data passed to the grader.

| Area | What to check |
| --- | --- |
| Instructions | Does the prompt state what is graded? Are required files, formats, constraints, and allowed tools clear? |
| Setup | Are inputs and dependencies available? Do paths, timeouts, services, and selected images match the task? |
| Correct work | Can valid answers fail because of formatting, overly narrow tests, or an unreliable judge? |
| Incorrect work | Can empty, hard-coded, partial, or otherwise invalid answers pass? Are all requested requirements checked? |
| Reward | Are weights, partial credit, tolerances, and score normalization correct? Are errors mistaken for wrong answers? |
| Exposure | Can the solver read answers, private tests, reference patches, hidden files, or useful Git history? Can it alter the grader or reward files? |
| External access | Do available tools, credentials, and network access allow forbidden answer lookup or model calls? |

Use these as checks, not assumptions that a task is defective. For example,
visible tests or internet access may be part of the intended task. Compare each
finding with the benchmark's stated rules.

## Check Verifiers behavior

- Rewards sum `score * weight`; metrics do not add to reward. `None` means a check did not produce a score.
- Scoring methods in the same group run concurrently. Check for one method depending on another's unfinished work.
- Judges have their own model and endpoint settings. Inspect their prompt, reference data, parsing, and recorded verdicts.
- Python hooks run on the evaluator. Runtime network restrictions do not cover them or tools in separate runtimes.
- Task data and `trace.info` are saved; `trace.state` is not. Check that the evidence needed to explain a score survives the run.
- For separate verification, follow the files collected from the solver, their restored paths, the verifier image, and the tests actually run.

See [tasksets](../../docs/v1/tasksets.md), [runtimes](../../docs/v1/runtimes.md),
and [Harbor](../../docs/v1/harbor.md) for these rules.

## Verify findings

For a claimed wrong grade, connect four pieces of evidence:

1. What the task requires, with a source link or exact excerpt.
2. What the agent produced, with a task/episode ID and saved reply or file.
3. What the grader actually checked and returned.
4. Why that result conflicts with the requirement.

For reward hacking, distinguish an available shortcut, an attempted shortcut,
and a shortcut shown to affect the reward. A suspicious command or an exposed
answer alone does not prove a cheated pass. A code defect can be confirmed
without proving that any saved run used it; report those claims separately.

When a runnable check is needed and within the requested scope, use an isolated
runtime and the smallest example that tests the claim. Check a known-good answer
and relevant bad answers, such as an empty output or a plausible wrong result.
Keep original task files and runs intact. Reading traces does not require
executing their commands.

`vf-validate` checks a known solution and the untouched task in separate runtimes.
Use `--only-noop` to score the untouched task alone. It flags rewards of at least
1.0 and can call configured judges. Inspect partial rewards as well; a score
below that threshold does not prove the grader is sound. Gold `unchecked` means
no verdict exists. Replay checks saved data but skips checks that need a runtime.
See [debug-environments](../debug-environments/SKILL.md)
for commands and limitations. If a necessary check has not run, state what is
still unknown.

## Report and fix

For each finding, give the affected task IDs, evidence, impact on scoring, and
smallest useful fix. Separate confirmed defects, confirmed reward effects, and
open questions. Report task defects and affected trials with their own counts;
one defective task may appear in many trials.

If fixes are requested, check the failing case and a valid case after the change.
Do not quietly change the benchmark's intended requirements. Use
[create-environments](../create-environments/SKILL.md) for implementation and
[evaluate-environments](../evaluate-environments/SKILL.md) for follow-up runs.
