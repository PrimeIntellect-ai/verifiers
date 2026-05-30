# hello-group-reward-v1

Deterministic v1 environment for exercising the group scoring boundary.

Each base task contains several candidate answers. `GroupRewardTaskset.init_group`
expands one base task into rollout-specific tasks, one candidate per rollout.
The harness program emits the assigned candidate without calling a model. After
all rollouts finish, the group stage:

- runs a group update that writes per-rollout ranks and group summaries;
- records group metrics;
- adds a relative group reward;
- writes explicit centered advantages;
- runs group cleanup.

This is meant as a compact reference for `@vf.reward(stage="group")` and
`@vf.advantage` behavior without model nondeterminism.
