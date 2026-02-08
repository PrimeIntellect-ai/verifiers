ENDPOINTS = {
    # allenai
    "olmo3-32b-t": {
        "model": "allenai/olmo-3-32b-think",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "olmo3-7b-i": {
        "model": "allenai/olmo-3-7b-instruct",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "olmo3-7b-t": {
        "model": "allenai/olmo-3-7b-think",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    # arcee
    "trinity-mini": {
        "model": "arcee/trinity-mini",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    # anthropic
    "haiku": {
        "model": "anthropic/claude-4.5-haiku",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "sonnet": {
        "model": "anthropic/claude-4.5-sonnet",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "opus": {
        "model": "anthropic/claude-4.5-opus",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    # google
    "gemini-2.5-flash": {
        "model": "google/gemini-2.5-flash",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "gemini-2.5-pro": {
        "model": "google/gemini-2.5-pro",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "gemini-3-flash": {
        "model": "google/gemini-3-flash",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "gemini-3-pro": {
        "model": "google/gemini-3-pro-preview",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "gemini-3-pro-exp": {
        "model": "google/gemini-3-pro-preview",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    # qwen
    "qwen3-30b-i": {
        "model": "qwen/qwen3-30b-a3b-instruct-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-30b-t": {
        "model": "qwen/qwen3-30b-a3b-thinking-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-235b-i": {
        "model": "qwen/qwen3-235b-a22b-instruct-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-235b-t": {
        "model": "qwen/qwen3-235b-a22b-thinking-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-vl-30b-i": {
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-vl-30b-t": {
        "model": "qwen/qwen3-vl-30b-a3b-thinking",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-vl-235b-i": {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "qwen3-vl-235b-t": {
        "model": "qwen/qwen3-vl-235b-a22b-thinking",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    # moonshot
    "kimi-k2": {
        "model": "moonshotai/kimi-k2-0905",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "kimi-k2-t": {
        "model": "moonshotai/kimi-k2-thinking",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    # openai
    "gpt-oss-120b": {
        "model": "openai/gpt-oss-120b",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "gpt-oss-20b": {
        "model": "openai/gpt-oss-20b",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "gpt-4.1-nano": {
        "model": "gpt-4.1-nano",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1": {
        "model": "gpt-4.1",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5-nano": {
        "model": "gpt-5-nano",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5-mini": {
        "model": "gpt-5-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5": {
        "model": "gpt-5",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5.1": {
        "model": "gpt-5.1",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-5.2": {
        "model": "gpt-5.2",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
    },
    # z-ai
    "glm-4.5": {
        "model": "z-ai/glm-4.5",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "glm-4.5-air": {
        "model": "z-ai/glm-4.5-air",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "glm-4.6": {
        "model": "z-ai/glm-4.6",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
    "glm-4.7": {
        "model": "z-ai/glm-4.7",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
    },
}
