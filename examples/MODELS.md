# Available models per endpoint

Each row's **Model ID** column is the literal string to put in `LLM_MODEL=`
or `EMBEDDING_MODEL=` for the matching `.env.*` template. Provider links
point to the canonical "available models" page at the upstream — those are
the source of truth for what's currently active.

## PNNL ai-incubator-api

For the `.env.pnnl`, `.env.bedrock.pnnl`, and `.env.pnnl.bedrock` examples.

Source: PNNL internal — ask your gateway admin for the live catalog.

### Chat

| Model ID | Notes |
|---|---|
| `claude-sonnet-4-20250514-v1-project` | Default — good balance |
| `claude-sonnet-4-5-20250929-v1-project` | Newer Sonnet |
| `claude-opus-4-20250514-v1-project` | Most capable Opus |
| `claude-opus-4-5-20251101-v1-project` | Newer Opus |
| `claude-haiku-4-5-20251001-v1-project` | Fast / cheap |
| `claude-3-5-haiku-20241022-project` | Older Haiku |
| `gpt-4o-project` | OpenAI GPT-4o |
| `gpt-4.1-project` | OpenAI GPT-4.1 |
| `gpt-5-project` | OpenAI GPT-5 |
| `gpt-5.1-project` | OpenAI GPT-5.1 |
| `gpt-5.2-project` | OpenAI GPT-5.2 |
| `o3-project` | OpenAI reasoning |
| `o3-mini-project` | OpenAI reasoning, smaller |
| `o4-mini-project` | OpenAI reasoning, smaller |
| `gemini-2.5-pro-project` | Google Gemini |
| `gemini-2.5-flash-project` | Google Gemini, fast |
| `grok-4-project` | xAI Grok |
| `grok-4-fast-reasoning-project` | xAI Grok with reasoning |
| `grok-4-fast-non-reasoning-project` | xAI Grok without reasoning |

### Embeddings

| Model ID | Dim |
|---|---|
| `text-embedding-3-small-project` | 1536 |
| `text-embedding-3-large-project` | 3072 |
| `gemini-embedding-001-project` | 768 |

---

## AWS Bedrock

For the `.env.bedrock`, `.env.bedrock.pnnl`, and `.env.pnnl.bedrock` examples.

Sources:
- Foundation models: <https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html>
- Cross-region inference profiles: <https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html>

> **Anthropic models on Bedrock require the `us.` cross-region inference
> profile prefix.** Plain `anthropic.claude-…-v1:0` returns "on-demand
> throughput isn't supported." Embedding models use the bare foundation
> id (no `us.` prefix); Cohere uses the profile id.

### Chat (Anthropic, cross-region inference profiles)

| Model ID | Notes |
|---|---|
| `us.anthropic.claude-opus-4-7` | Newest Opus |
| `us.anthropic.claude-opus-4-6-v1` | |
| `us.anthropic.claude-opus-4-5-20251101-v1:0` | |
| `us.anthropic.claude-opus-4-1-20250805-v1:0` | |
| `us.anthropic.claude-sonnet-4-6` | Newest Sonnet |
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Fast / cheap |
| `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | |
| `us.anthropic.claude-3-5-haiku-20241022-v1:0` | Legacy |
| `us.anthropic.claude-3-haiku-20240307-v1:0` | Legacy |

### Chat (Amazon Nova, cross-region inference profiles)

| Model ID | Notes |
|---|---|
| `us.amazon.nova-premier-v1:0` | Largest Nova |
| `us.amazon.nova-pro-v1:0` | |
| `us.amazon.nova-lite-v1:0` | |
| `us.amazon.nova-2-lite-v1:0` | |
| `us.amazon.nova-micro-v1:0` | Smallest |

### Embeddings

| Model ID | Dim |
|---|---|
| `amazon.titan-embed-text-v2:0` | 1024 default (configurable: 256/512/1024) |
| `us.cohere.embed-v4:0` | 1024 |

---

## Anthropic API

For the `.env.anthropic.openai` example.

Source: <https://docs.anthropic.com/en/docs/about-claude/models/all-models>

> Use the bare model ID — LiteLLM's anthropic provider doesn't need a
> prefix. **Do NOT include `/v1` in `LLM_BASE_URL`.**

### Chat

| Model ID | Notes |
|---|---|
| `claude-opus-4-5-20251101` | Latest Opus |
| `claude-opus-4-1-20250805` | |
| `claude-sonnet-4-5-20250929` | Latest Sonnet |
| `claude-haiku-4-5-20251001` | Fast / cheap |
| `claude-3-7-sonnet-20250219` | |
| `claude-3-5-haiku-20241022` | Legacy |

### Embeddings

Anthropic does not provide an embeddings endpoint. Pair with OpenAI,
Voyage, Cohere, or Bedrock embeddings instead.

---

## OpenAI API

For the embedding side of `.env.anthropic.openai`. Also usable as the
chat side if you set `LLM_PROVIDER=openai` with `LLM_BASE_URL=https://api.openai.com/v1`.

Source: <https://platform.openai.com/docs/models>

### Chat

| Model ID | Notes |
|---|---|
| `gpt-5` | |
| `gpt-5-mini` | |
| `gpt-5-nano` | |
| `gpt-4.1` | |
| `gpt-4o` | |
| `gpt-4o-mini` | |
| `o3` | Reasoning |
| `o3-mini` | Reasoning, smaller |
| `o4-mini` | Reasoning, smaller |

### Embeddings

| Model ID | Dim |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 (legacy) |
