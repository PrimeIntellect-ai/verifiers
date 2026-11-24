# web-search-interception

Inspects a buffered Codex response, including provider-native web-search items,
and uses `@vf.stop` when the search material contains a chosen phrase.

Codex's web search is off by default. Turn it on, against a model whose provider
serves the Responses `web_search` tool:

```
vf-run web_search_interception -m <model> --env.agent.harness.web-search live
```
