# Release Notes - v0.1.3.post0

Minor release for miscellaneous fixes + small API tweaks which should not impact the vast majority of users, beyond addressing bugs. Full notes + credits deferred to next proper version release.

Changelog:
```
* 5d887a9 (HEAD -> main) docs -> dev deps
* 9be7d22 (origin/main) Fix setting log level globally (#296)
* 5342a18 Rename `max_concurrent_requests` to `max_concurrent` (#295)
* 9d962b0 Fix `max_concurrent_requests` in eval script and also use for rollout scoring (#294)
* bfc98e3 hotfix for json-serialized info
* b314ce7 disable max turns default (#292)
* d66990c readme
* 061b28a version
* 631257e post0 version
* 4f2a71c t-e pyproj bump
* 5970f67 toxicity_explanation hotfix
* 4558ffe fix: add robust function schema parsing (#285)
* cca161f Math python tweak (#286)
* 94fef56 fix markuperror in completion (#284)
* 4daa4b3 chore(eval): add logging throughout evaluation script for better traceability (#262)
* 2871822 feat(verifiers): add MathRubric to verifiers module (#263)
* 292214b docs(rubric): add documentation for passing class objects to reward functions (#269)
* bfbb311 docs(env): clarify optional answer/info fields and evaluation behavior (#268)
* 6675c8b answer + info both optional (#282)
* 5305b16 fix(tui): escape user content to prevent markup injection issues (#273)
* d64d701 Fix missing parser parameter in Rubric instances across environments (#276)
* fb1b4c1 fix
* b0e2df3 readme
* 615ab08 Fix eval saving failing on `-n -1` (#255)
* 85ae8e4 detect when tool_calls is a list of JSON strings (#250)
* 2106820 (tag: v0.1.3) Release version 0.1.3
```