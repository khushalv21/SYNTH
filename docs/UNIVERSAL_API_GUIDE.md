# Universal API Configuration Guide

Synth's `UniversalAPIAuthenticator` can connect to **any** HTTP API — OpenAI, Anthropic, Ollama, or your own custom endpoint. This guide shows you how.

---

## Quick Setup

### 1. Create a `.env` file

Copy the template:

```bash
cp .env.example .env
```

### 2. Set your credentials

```env
SYNTH_API_BASE_URL=https://api.openai.com/v1/chat/completions
SYNTH_API_KEY=sk-your-key-here
SYNTH_API_MODEL=gpt-4o
```

### 3. Run

```bash
synth image.png --engine api
```

---

## Provider Configurations

### OpenAI (GPT-4o, GPT-4, GPT-3.5)

```env
SYNTH_API_BASE_URL=https://api.openai.com/v1/chat/completions
SYNTH_API_KEY=sk-your-key-here
SYNTH_API_MODEL=gpt-4o
SYNTH_PAYLOAD_MAP=./config/payload_openai.json
```

### Anthropic (Claude)

```env
SYNTH_API_BASE_URL=https://api.anthropic.com/v1/messages
SYNTH_API_KEY=sk-ant-your-key-here
SYNTH_API_MODEL=claude-sonnet-4-20250514
SYNTH_PAYLOAD_MAP=./config/payload_anthropic.json
```

> **Note:** Anthropic uses `x-api-key` header without a `Bearer` prefix. The `payload_anthropic.json` config handles this automatically.

### Ollama (Local LLM)

```env
SYNTH_API_BASE_URL=http://localhost:11434/api/chat
SYNTH_API_KEY=not-needed
SYNTH_API_MODEL=llama3
```

Create `config/payload_ollama.json`:

```json
{
    "payload_template": {
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI content detector. Respond with JSON: {\"score\": 0.0-1.0, \"verdict\": \"human|ai|mixed\", \"reasoning\": \"...\"}"
            },
            {
                "role": "user",
                "content": "{text}"
            }
        ],
        "stream": false
    },
    "score_path": "message.content",
    "label_path": "message.content",
    "reasoning_path": "message.content"
}
```

```env
SYNTH_PAYLOAD_MAP=./config/payload_ollama.json
```

### Custom Detection API

If you have a dedicated detection endpoint that returns structured JSON:

```env
SYNTH_API_BASE_URL=https://your-api.com/v1/detect
SYNTH_API_KEY=your-key
```

Create `config/payload_custom.json`:

```json
{
    "payload_template": {
        "text": "{text}",
        "options": {
            "detailed": true
        }
    },
    "score_path": "result.ai_probability",
    "label_path": "result.classification",
    "reasoning_path": "result.explanation"
}
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|:---:|---|
| `SYNTH_API_BASE_URL` | ✅ | Full API endpoint URL |
| `SYNTH_API_KEY` | ✅ | Authentication key |
| `SYNTH_API_MODEL` | ❌ | Model identifier |
| `SYNTH_PAYLOAD_MAP` | ❌ | Path to JSON payload mapping file |

---

## Payload Mapping Format

The JSON config file has two sections: **request shape** and **response parsing**.

### Request: `payload_template`

A JSON object sent as the POST body. Use `{text}` as a placeholder — it gets replaced with the actual input text at runtime.

```json
{
    "payload_template": {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Analyse this: {text}"}
        ]
    }
}
```

The `{text}` substitution works at any nesting depth — strings, arrays, nested objects.

### Response: Dot-Notation Paths

Tell Synth where to find the result in the API response using dot notation. Array indices use numbers.

| Field | Description | Example |
|---|---|---|
| `score_path` | Path to the score (or JSON string containing it) | `choices.0.message.content` |
| `label_path` | Path to the verdict label | `result.label` |
| `reasoning_path` | Path to the explanation | `result.reasoning` |

**Example:** For an OpenAI response:

```json
{
    "choices": [
        {
            "message": {
                "content": "{\"score\": 0.85, \"verdict\": \"ai\", \"reasoning\": \"...\"}"
            }
        }
    ]
}
```

The path `choices.0.message.content` resolves to the JSON string, which Synth then parses automatically.

### Auth Header Overrides

Some APIs use non-standard auth headers:

```json
{
    "auth_header": "x-api-key",
    "auth_prefix": ""
}
```

| Field | Default | Description |
|---|---|---|
| `auth_header` | `Authorization` | HTTP header name |
| `auth_prefix` | `Bearer` | Prefix before the key (empty string for no prefix) |

---

## Programmatic Usage

```python
from synth.core.auth import APIEndpointConfig, UniversalAPIAuthenticator

# From .env
auth = UniversalAPIAuthenticator()

# From explicit config
cfg = APIEndpointConfig(
    base_url="https://api.openai.com/v1/chat/completions",
    api_key="sk-...",
    model="gpt-4o",
)
auth = UniversalAPIAuthenticator(config=cfg)

# From JSON file
cfg = APIEndpointConfig.from_json("config/payload_openai.json")

# Use as context manager
with UniversalAPIAuthenticator(config=cfg) as auth:
    result = auth.detect("Text to analyse...")
    print(result.verdict)
```
