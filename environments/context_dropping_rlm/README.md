# context-dropping-rlm

### Overview
- **Environment ID**: `context-dropping-rlm`
- **Short description**: Tests context dropping in RLMEnv. Model must find facts across many files while managing its context window.
- **Tags**: rlm, context-management, multi-turn

### Task

The model receives N files (default 15), each containing one key-value fact buried in filler text. The question asks for a subset of those facts (default 5). The filler text is large enough that reading all files fills the context window, forcing the model to use `remove_conversation_turns` to drop old turns and keep working.

The model can read `.messages` to recover dropped context and `.summaries` to review past summaries.

### Metrics

| Metric | Meaning |
| ------ | ------- |
| partial_match_reward | Fraction of target facts found |
| exact_match_reward | 1.0 only if all target facts found |
| context_drop_count | Number of times context was dropped |
| context_total_turns_dropped | Total turns dropped |
| context_drop_mean_remaining_turns | Average turns remaining after each drop |
| context_drop_mean_turns_between | Average turns between consecutive drops |

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| num_samples | int | 5 | Number of dataset samples |
| num_files | int | 15 | Files per sample |
| num_target_facts | int | 5 | Facts the question asks about |
| filler_words_per_file | int | 400 | Filler words per file |
| allow_context_dropping | bool | True | Enable context dropping tool |
| min_turns_in_context | int | 3 | Minimum turns to keep |
| max_turns | int | 30 | Maximum REPL turns |
