# sft-replay

Replay stored chat transcripts into Verifiers trajectory steps without making
model requests.

Each local example is one JSON object under `data/`:

```json
{
  "messages": [
    {"role": "user", "content": "Reverse abc."},
    {"role": "assistant", "content": "cba"}
  ]
}
```

`messages` is a JSON array of chat message objects. Each message must have a
string `role`; all other fields are preserved.

The environment uses `ReplayTaskset` to load transcript rows and
`ReplayHarness` to replay each assistant message as one trajectory step with
`tokens=None`. Non-assistant messages may appear before, between, or after
assistant messages.
