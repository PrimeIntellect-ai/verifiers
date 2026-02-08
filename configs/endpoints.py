ENDPOINTS = {
    # allenai
    "olmo3-32b-t": {
        "model": "allenai/olmo-3-32b-think",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "olmo3-7b-i": {
        "model": "allenai/olmo-3-7b-instruct",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "olmo3-7b-t": {
        "model": "allenai/olmo-3-7b-think",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    # arcee
    "trinity-mini": {
        "model": "arcee/trinity-mini",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    # anthropic
    "haiku": {
        "model": "claude-haiku-4-5",
        "url": "https://api.anthropic.com",
        "key": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
    "sonnet": {
        "model": "claude-sonnet-4-5",
        "url": "https://api.anthropic.com",
        "key": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
    "opus": {
        "model": "claude-opus-4-5",
        "url": "https://api.anthropic.com",
        "key": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
    # deepseek
    "deepseek-chat": {
        "model": "deepseek-chat",
        "url": "https://api.deepseek.com/v1",
        "key": "DEEPSEEK_API_KEY",
        "client_type": "openai",
    },
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
        "url": "https://api.deepseek.com/v1",
        "key": "DEEPSEEK_API_KEY",
        "client_type": "openai",
    },
    # google
    "gemini-2.5-flash": {
        "model": "google/gemini-2.5-flash",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "gemini-2.5-pro": {
        "model": "google/gemini-2.5-pro",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "gemini-3-flash": {
        "model": "google/gemini-3-flash",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "gemini-3-pro": {
        "model": "google/gemini-3-pro-preview",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "gemini-3-pro-exp": {
        "model": "google/gemini-3-pro-preview",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    # qwen
    "qwen3-30b-i": {
        "model": "qwen/qwen3-30b-a3b-instruct-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-30b-t": {
        "model": "qwen/qwen3-30b-a3b-thinking-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-235b-i": {
        "model": "qwen/qwen3-235b-a22b-instruct-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-235b-t": {
        "model": "qwen/qwen3-235b-a22b-thinking-2507",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-vl-30b-i": {
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-vl-30b-t": {
        "model": "qwen/qwen3-vl-30b-a3b-thinking",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-vl-235b-i": {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "qwen3-vl-235b-t": {
        "model": "qwen/qwen3-vl-235b-a22b-thinking",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    # moonshot
    "kimi-k2": {
        "model": "moonshotai/kimi-k2-0905",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "kimi-k2-t": {
        "model": "moonshotai/kimi-k2-thinking",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    # openai
    "gpt-oss-120b": {
        "model": "openai/gpt-oss-120b",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "gpt-oss-20b": {
        "model": "openai/gpt-oss-20b",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "gpt-4.1-nano": {
        "model": "gpt-4.1-nano",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-4.1": {
        "model": "gpt-4.1",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-5-nano": {
        "model": "gpt-5-nano",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-5-mini": {
        "model": "gpt-5-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-5": {
        "model": "gpt-5",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-5.1": {
        "model": "gpt-5.1",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    "gpt-5.2": {
        "model": "gpt-5.2",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY",
        "client_type": "openai",
    },
    # z-ai
    "glm-4.5": {
        "model": "z-ai/glm-4.5",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "glm-4.5-air": {
        "model": "z-ai/glm-4.5-air",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "glm-4.6": {
        "model": "z-ai/glm-4.6",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
    "glm-4.7": {
        "model": "z-ai/glm-4.7",
        "url": "https://api.pinference.ai/api/v1",
        "key": "PRIME_API_KEY",
        "client_type": "openai",
    },
}
