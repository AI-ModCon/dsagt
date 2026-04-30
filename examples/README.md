# Example `.env` files

Reference templates for the upstream provider combinations DSAGT supports.
Each file is a complete `.env` you can copy to `<repo>/.env` and fill in
your own API keys (the templates ship with empty `*_API_KEY=` lines — fill
them in, don't commit).

## Naming convention

`.env.<llm>` when both endpoints share a provider.
`.env.<llm>.<embedding>` when they differ — the first segment names the
chat-completion provider, the second names the embedding provider.

| File | Chat | Embeddings |
|---|---|---|
| `.env.bedrock` | AWS Bedrock | AWS Bedrock |
| `.env.pnnl` | PNNL ai-incubator-api | PNNL ai-incubator-api |
| `.env.anthropic.openai` | Anthropic API | OpenAI API |
| `.env.bedrock.pnnl` | AWS Bedrock | PNNL ai-incubator-api |
| `.env.pnnl.bedrock` | PNNL ai-incubator-api | AWS Bedrock |

## Usage

```bash
cp examples/.env.pnnl .env
# edit .env — fill in LLM_API_KEY and EMBEDDING_API_KEY
dsagt smoke-test --agent claude
```

## Agent compatibility

All five agents (`claude`, `goose`, `cline`, `roo`, `codex`) work
with any OpenAI-wire-compatible upstream (PNNL, OpenAI directly, etc.).

**Codex does not work with AWS Bedrock chat.** Codex 0.125 wraps MCP
tools in a non-standard `{type: namespace, ...}` schema that Bedrock's
Anthropic Messages adapter doesn't unpack — the model sees opaque
namespace stubs and never invokes your registered tools. Use a different
agent for Bedrock chat. The `.env.pnnl.bedrock` configuration (PNNL
chat + Bedrock embeddings) is the workaround if you specifically want
Bedrock embeddings with codex.

## Gotchas

- **Anthropic API base URL must NOT include `/v1`.** LiteLLM auto-appends
  `/v1/messages` and only checks for that exact suffix; a trailing `/v1`
  produces `/v1/v1/messages` and 404s.
- **AWS Bedrock bearer-token API keys expire after 12 hours.** Regenerate
  from the AWS console when you start seeing 403 "Bearer Token has
  expired" errors.
- **Bedrock Anthropic models require a cross-region inference profile id**
  (the `us.` prefix). The bare model id returns "on-demand throughput
  isn't supported."
- **Embedding dimension mismatch on switch.** Each collection's
  `route.json` pins the embedder used at ingest time. Switching from
  text-embedding-3-small (1536-dim) to titan-embed-text-v2 (1024-dim)
  doesn't migrate prior collections — you'll need to re-ingest. Search
  on existing collections continues to work via the persisted route.
