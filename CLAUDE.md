# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repository exposes Notion AI as a local API proxy with two compatibility surfaces:
- OpenAI-compatible endpoints under `/v1` for standard chat clients.
- Anthropic-compatible `/v1/messages` for Claude Code and Anthropic SDK clients.

The service supports three runtime modes controlled by `APP_MODE` in `.env`:
- `lite`: stateless mode, only the latest user prompt is forwarded.
- `standard`: full client-supplied history is forwarded, with thinking/search output passthrough.
- `heavy`: server-managed conversation state backed by SQLite, including sliding-window memory and summarization of older rounds.

## Common commands

### Environment setup
```bash
pip install -r requirements.txt
```

### Run the API server
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

### Run the CLI chat entrypoint
```bash
python main.py
```

### Syntax check
```bash
python -m compileall app main.py
```

### Health and model checks
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

### Run a single targeted test
There is currently no `tests/` directory in this repository. If tests are added later, prefer targeted pytest execution such as:
```bash
pytest path/to/test_file.py
pytest path/to/test_file.py -k test_name
```

## Required configuration

The app reads configuration from `.env` at import time via `app/config.py`.

Minimum required settings:
- `NOTION_ACCOUNTS`: JSON array of Notion account credentials. Each entry must at least include `token_v2`, `space_id`, and `user_id`.

Frequently used optional settings:
- `APP_MODE`: `lite`, `standard`, or `heavy`.
- `API_KEY`: enables Bearer auth for `/v1*` routes when set.
- `HOST` / `PORT`: server bind settings.
- `DB_PATH`: SQLite path used by heavy mode.
- `SILICONFLOW_API_KEY`: enables summarization for heavy mode compression.
- `ALLOWED_ORIGINS`: comma-separated CORS allowlist.

## High-level architecture

### Request flow
1. `app/server.py` creates the FastAPI app, installs CORS, rate limiting, structured logging, API key auth, and mounts routers.
2. During lifespan startup, it initializes a shared `AccountPool`; in `heavy` mode it also creates a `ConversationManager`.
3. Route handlers in `app/api/chat.py` and `app/api/anthropic.py` normalize incoming protocol payloads into the repository’s internal transcript format.
4. `AccountPool` selects a Notion account via round-robin and temporarily cools down failing accounts.
5. `NotionOpusAPI` in `app/notion_client.py` converts internal transcript config blocks to Notion-specific model/thread payloads and streams NDJSON from Notion’s upstream endpoint.
6. `app/stream_parser.py` parses upstream patches into normalized events such as `content`, `thinking`, `search`, and `final_content`.
7. API routes translate those normalized events back into either OpenAI SSE/JSON or Anthropic SSE/JSON responses.

### Core modules
- `app/server.py`: app assembly, middleware, startup state, `/health`, and router registration.
- `app/api/chat.py`: OpenAI-style `/v1/chat/completions`; contains most streaming adaptation logic, recall detection, and heavy-mode integration.
- `app/api/anthropic.py`: Anthropic `/v1/messages` compatibility layer; converts Anthropic request/response formats to and from the internal stream abstraction.
- `app/notion_client.py`: upstream Notion transport, thread lifecycle, request payload shaping, and retryable upstream error classification.
- `app/account_pool.py`: multi-account round-robin selection and cooldown-based failover.
- `app/conversation.py`: heavy-mode persistence and memory model using SQLite (`messages`, `sliding_window`, `compressed_summaries`, `full_archive`, and conversation metadata including `thread_id`). Also builds internal transcript blocks.
- `app/stream_parser.py`: decodes Notion NDJSON patch streams and strips internal markup noise before exposing normalized stream events.
- `app/model_registry.py`: source of truth for exposed model IDs, Notion upstream model mappings, and thread type selection.
- `app/summarizer.py`: optional SiliconFlow-backed summarization used when heavy mode compresses older turns.
- `app/prompt_injection.py`: refusal detection and prompt-shaping utilities used to work around upstream model/tool restrictions.
- `main.py`: local terminal chat client reusing `ConversationManager` and `NotionOpusAPI` outside FastAPI.

## Mode-specific behavior

### Lite mode
- No server-side memory.
- Only the last user message is sent upstream.
- Best for simple stateless usage and lowest complexity.

### Standard mode
- Uses the full request history supplied by the client.
- Preserves thinking/search output in API adaptation layers.
- Does not persist long-term history server-side.

### Heavy mode
- Creates `ConversationManager` on startup and persists conversations to SQLite.
- Maintains both a recent sliding window and a long-term archive.
- Older rounds can be summarized through SiliconFlow when `SILICONFLOW_API_KEY` is configured.
- Conversation records also persist Notion `thread_id` so upstream threads can be reused for continuity.

## Data and protocol notes

- OpenAI-compatible routes live under `/v1/chat/completions` and `/v1/models`.
- Anthropic-compatible routes live under `/v1/messages`.
- `/health` reports uptime plus account pool availability.
- The model IDs exposed to clients are mapped in `app/model_registry.py`; changing public model names requires keeping the Notion mapping in sync.
- Gemini-family models use `markdown-chat` thread type, while the rest use `workflow` thread type.

## Operational constraints worth remembering

- `app/config.py` loads `.env` immediately on import, so missing `NOTION_ACCOUNTS` breaks startup early.
- This project currently has no formal automated test suite; use syntax checks and endpoint smoke tests unless/until tests are introduced.
- `data/conversations.db` is part of the runtime state for heavy mode and reflects local conversation persistence.
- README includes the supported setup flow for extracting Notion credentials via `scripts/extract_notion_info.js`.
