# Verifiers v0.1.13.dev2 Release Notes

*Date:* 04/19/2026

## Highlights since v0.1.13.dev1

- Added richer token usage reporting, including final-context token metrics, updated displays, and API docs.
- Expanded `vf-tui` compare mode so you can inspect any numeric metric with inline selection and adaptive bucketing.
- Improved composable/RLM harness integration with harness-owned upload dirs, cached local RLM checkouts, and auto-registered tool monitoring from `harness.tool_names`.
- Surfaced CLI agent crashes as infra errors even after prior turns, and now include full traces in agent error logs for debugging.
- Removed dead RLM tool config constants from the composable harness exports.

## Changes included in v0.1.13.dev2 (since v0.1.13.dev1)

### Features and enhancements

- Better token count metrics (#1108)
- vf-tui: compare for all metrics (#1117)
- harness.get_upload_dirs; reduce rlm github requests (#1178)
- tool-env: ToolMonitorRubric takes tool_names instead of tools (#1179)
- feat: auto-register ToolMonitorRubric from harness.tool_names (#1181)
- feat: include full trace in agent error logs (#1185)

### Fixes and maintenance

- cli-agent: surface agent crashes as infra errors after any turn (#1177)
- Remove dead RLM tool config constants (#1189)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev1...v0.1.13.dev2
