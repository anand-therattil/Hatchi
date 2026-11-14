# Hatchi — personalized AI chat with long-term memory

A small chat service that answers with an LLM, follows up on recent conversation, **remembers useful
facts about each user across sessions**, and retrieves only the memories relevant to the current question.
Built for the MyNaksh "Personalized AI Chat" assignment (FastAPI · Redis · MongoDB · Chroma · vLLM).

## Architecture

```
            ┌────────────── FastAPI ──────────────┐
 WebSocket  │  /ws/chat   (streaming, session end) │
 HTTP       │  POST /chat, /users/.../end, /memories│
            └───────────────┬──────────────────────┘
                            │  ChatService
        ┌───────────────────┼──────────────────────────────┐
        │ request path      │ background path (per user)   │ session end
        ▼                   ▼                              ▼
  Retriever ──► Prompt ──► LLM          Extractor ─► MemoryManager     Summarizer (final)
  (Redis recent turns,                  (gate→rules→LLM)  upsert/update      │
   Mongo by category,                                      │                 ▼
   Chroma semantic)                                        ▼          Mongo + Chroma
                                                    Mongo (truth) + Chroma (index)

 Redis   short-term: last N messages + rolling summary per session
 MongoDB long-term:  user_memories (one doc per fact), session_summaries
 Chroma  semantic index over memories and session summaries (Granite multilingual embeddings)
 vLLM    Llama-3.2-3B-Instruct (or Qwen2.5-1.5B-Instruct) via its OpenAI-compatible API — behind an
         `LLMClient` interface, so Ollama / a mock are drop-in alternatives
```

### One chat turn
1. Load the session's recent messages and rolling summary from **Redis**.
2. **Retrieve** relevant memories and past-session summaries (below).
3. Build the prompt: system rules + selected memories + earlier-session notes + rolling summary + recent turns + message.
4. Call the LLM (streamed over WebSocket, plain over HTTP). If it is down, return a graceful fallback and carry on.
5. Append the turn to Redis and **return** `response` + `context_used`.
6. In the background (serialised per user): extract/update memories, and fold old turns into the rolling summary
   if the window overflowed. The reply never waits on this.

### Memory: what is saved
Not every message — only durable facts. A pipeline decides:
1. **Gate** — only declarative, first-person sentences are considered. Questions and requests
   ("What should I focus on for my career?", "Tell me about Saturn") are dropped.
2. **Rules** — high-precision regexes for the obvious cases (language preference, job-switch plans, current job,
   location, birth details, relationship status, interests, …).
3. **LLM** — fills gaps using JSON-schema-constrained output. Its output is validated: the key must be in the
   closed vocabulary (`app/memory/schema.py`) and the value must share a word with what the user said
   (hallucination guard). Rules win over the LLM for the same slot.

The vocabulary is closed on purpose: small models invent inconsistent key names, and a stable key is what makes
corrections a simple upsert.

### Memory: updates and corrections
| Kind | Example key | Behaviour |
|---|---|---|
| slot | `career_goal`, `language_preference` | new value replaces the old; old value archived in `history`, `version` bumped |
| list | `interests` | values merged and de-duplicated |
| fact | `fact_<hash>` | near-duplicate notes (cosine ≥ 0.9) update the existing note |

An *inferred* value (confidence 0.6) never overwrites an *explicit* one (0.9); an explicit statement always wins.
"I'm not planning to switch jobs anymore" **deletes** `career_goal`. MongoDB is written first; if the Chroma write
fails the memory is flagged `indexed=false` and re-indexed on next startup.

### Memory: retrieval ("context relevance")
Three sources, three mechanisms:
- **Current session** — no retrieval; recent turns + rolling summary are in the prompt (this is what makes
  "Why do you say that?" work; short follow-ups also inherit the previous turn's topic and query).
- **Long-term facts** — candidates = *category match* (keyword stems, English + Hindi/Hinglish) ∪ *semantic search*
  (Chroma, filtered by `user_id`). A memory is included if it is an always-on standing preference
  (`language_preference`), **or** its category matches the question, **or** it is a strong semantic hit:
  similarity ≥ `MEMORY_MIN_SIMILARITY` **and** within `SEMANTIC_MARGIN` of the best hit for that question.
  Then ranked by `similarity + category boost + confidence` and capped at `MEMORY_TOP_K`.
  A career question therefore gets career goals, not "interests: cricket".
  "What do you remember about me?" (no category) returns everything up to `MEMORY_RECALL_MAX`.
- **Past sessions** — semantic search over finished-session summaries, excluding the live session, above a threshold.

**Thresholds must be calibrated per embedding model.** Real cosine scores are compressed: in the first live run Granite-97M
scored a relevant memory ~0.90 and an unrelated one ~0.79, so a naive 0.45 cut-off let "interests: cricket" into a career
answer. Hence the high floors plus the relative margin. The defaults are provisional; run
`python -m scripts.calibrate_similarity` to print scores for labelled English/Hindi/Hinglish cases, sweep thresholds, and
suggest `.env` values (edit the cases in that file to match your users' phrasing).

`context_used` lists exactly what reached the LLM (`career_goal`, `session_summary:<id>`, `conversation_summary`, …).
Send `"debug": true` to also get scores and the list of excluded memories.

### Summarisation
- **Rolling (mid-session):** when the Redis window exceeds `HISTORY_MAX_MESSAGES`, the oldest `HISTORY_FOLD_COUNT`
  messages are folded into a running summary. Not per message — that wastes LLM calls and degrades a small model.
- **Final (session end):** `POST /users/{u}/sessions/{s}/end`, a WebSocket `{"type":"end"}`, a WebSocket drop (after a
  grace period, unless the client reconnects), or the idle sweeper (`SESSION_IDLE_SECONDS`). The summary + topics go to
  MongoDB and are embedded into Chroma; the Redis state is cleared.
- Both fall back to a deterministic heuristic if the LLM is unavailable.

## Quick start

```bash
# 1. infrastructure: Redis, MongoDB, Chroma ...
docker compose up -d
#    ... plus vLLM on a Linux machine with an NVIDIA GPU (needs Docker + NVIDIA Container Toolkit):
cp .env.example .env             # set HF_TOKEN (Llama is gated: accept its license on Hugging Face first)
docker compose --profile gpu up -d
docker compose logs -f vllm      # first start downloads the model; wait for "Application startup complete"
curl localhost:8002/v1/models    # vLLM is ready when this lists the model

# 2. no GPU box / not using the compose service? Run a model server yourself instead:
#    - vLLM directly:   vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8002 --max-model-len 4096
#    - Apple Silicon:   install the vllm-metal plugin (docs.vllm.ai/projects/vllm-metal), serve an MLX build
#                       from the mlx-community org, and set VLLM_MODEL to that name
#    - Ollama:          ollama pull llama3.2:3b, then LLM_BACKEND=ollama in .env
#    The ungated alternative is Qwen/Qwen2.5-1.5B-Instruct (set VLLM_MODEL to it).

# 3. app
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cp -n .env.example .env
uvicorn app.main:app --reload    # http://localhost:8000/docs

# 4. chat from the terminal (WebSocket)
python cli.py --user user-123
```

The first request downloads the Granite embedding model (`ibm-granite/granite-embedding-97m-multilingual-r2`) from
Hugging Face. Check its model card for any query/document prefix convention and set `EMBEDDING_QUERY_PREFIX` /
`EMBEDDING_DOC_PREFIX` if required.

**No Docker / no model?** Run entirely in-process with a mock LLM:
```bash
STORAGE_BACKEND=memory VECTOR_BACKEND=ephemeral LLM_BACKEND=mock EMBEDDING_BACKEND=fake uvicorn app.main:app
```

## API

### `POST /chat`
```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "user_id": "user-123", "session_id": "session-A",
  "message": "I'"'"'m planning to switch jobs next year, and I prefer advice in Hindi."}'
```
```json
{"user_id":"user-123","session_id":"session-A","response":"…","context_used":[],"degraded":false,"debug":null}
```
Later, same session:
```json
{"user_id":"user-123","session_id":"session-A","response":"…","context_used":["language_preference","career_goal"],"degraded":false,"debug":null}
```
A new conversation, asking `"What do you remember about my career plans?"` with `"debug": true`:
```json
{
  "user_id": "user-123", "session_id": "session-B", "response": "…",
  "context_used": ["language_preference", "career_goal"],
  "degraded": false,
  "debug": {
    "categories": ["career"], "followup": false, "recall_intent": true, "considered": 2, "excluded": [],
    "memory_scores": {
      "language_preference": {"score": 0.045, "sim": 0.0,    "why": ["always_on"]},
      "career_goal":         {"score": 0.7632, "sim": 0.2182, "why": ["category"]}
    }
  }
}
```
(`response` text depends on the model; the other fields above are real output from a run with the mock LLM.)

Validation: `user_id`/`session_id` must match `[A-Za-z0-9_.@-]{1,128}`; `message` 1–4000 chars → otherwise `422`.

### Other endpoints
| | |
|---|---|
| `POST /users/{user_id}/sessions/{session_id}/end` | finalise a session → `{"ended": true, "summary": {…}}` |
| `GET /users/{user_id}/memories` | inspect stored memories (value, confidence, version, history) |
| `DELETE /users/{user_id}/memories/{key}` | forget a memory |
| `GET /health` | liveness |

### `WS /ws/chat?user_id=…&session_id=…` (`session_id` optional → generated)
```
→ {"type":"message","text":"What should I focus on for my career?"}
← {"type":"session","session_id":"…"}                 (on connect)
← {"type":"token","text":"Based "} …                   (streamed)
← {"type":"done","response":"…","context_used":["career_goal","language_preference"],"degraded":false}
→ {"type":"end"}
← {"type":"ended","summary":{"session_id":"…","summary":"…","topics":["career"]}}
```
State lives in Redis, so reconnecting with the same `session_id` resumes the conversation.

## Tests
```bash
pytest -q                  # 109 unit/behaviour tests, ~2s, no Docker / model / network needed
pytest -m integration -v   # 10 live-stack tests: real vLLM + Redis + MongoDB + Chroma + Granite
```
**Unit/behaviour tests** run the real service graph with a mock LLM, in-memory stores, an ephemeral Chroma and a
deterministic fake embedder (`tests/`):
- `test_chat_flow` - end-to-end through the HTTP API: send → response; memory saved in one session and retrieved in
  another; follow-up resolved from recent history; **irrelevant memories excluded** from the LLM prompt; corrections
  update in place; withdrawn plans deleted; per-user isolation; rolling summary; final-summary recall; LLM-outage
  fallback; idle sweeper; request validation.
- `test_memory_manager` - list merge, explicit-beats-inferred, history cap, fact de-duplication, invalid ops,
  one failing op not losing the rest, re-indexing after a Chroma outage, deletes.
- `test_retrieval` - always-on preference, category vs semantic inclusion, follow-up topic inheritance, "what do you
  remember" caps, stale index entries, user/session isolation, past-session thresholds.
- `test_services` - summariser fallbacks, empty/reused sessions, concurrent messages not losing updates, interrupted
  streams, prompt assembly.
- `test_units`, `test_api_and_rules_extra` - extraction gate and rules (English/Hinglish), LLM-output guards, keyword
  detection (incl. Devanagari), Redis (`fakeredis`) and Mongo (`mongomock-motor`) stores, vLLM and Ollama clients
  against fake HTTP servers, remaining endpoints.
- `test_websocket` - streaming, end-of-session, bad input, invalid ids, disconnect grace period.

**Integration tests** (`tests/integration/`) are skipped by default and skip cleanly, with the reason, if a service is
unreachable. They read `.env`, use a throw-away MongoDB database and Chroma prefix, and clean up. Besides the full
flows (memory recall across sessions, correction, session summary recall, streaming), they check the things the mocks
cannot: the model returns schema-valid JSON, the LLM can extract a memory no rule covers (`health_focus`), and Granite
ranks a career memory above a hobby for English and Hindi questions - that test prints the similarity scores, which is
what to use when tuning `MEMORY_MIN_SIMILARITY` and the summary thresholds.

## Design decisions and trade-offs
- **Closed memory vocabulary** over free-form keys: reliable updates with small models, at the cost of coverage
  (`other_fact` is the escape hatch).
- **Rules + LLM** for extraction: rules are deterministic and testable; the LLM catches what regexes miss.
  The extractor sends the user's existing memories to the LLM so it can express corrections.
- **Hybrid retrieval** (category + semantic): keywords are precise for obvious topics, embeddings handle paraphrase
  and Hindi/Hinglish. Thresholds are in `.env` and should be tuned against the real embedding model.
- **Background post-processing**: low latency, but the next message can arrive before the previous one's memory
  is written (a few seconds with a 3B model on CPU).
- **MongoDB as truth, Chroma as a rebuildable index**: Chroma can be wiped and rebuilt; `indexed` flags and
  `reindex_pending()` cover failed writes.
- **HTTP + WebSocket** share one `ChatService`. HTTP is stateless and is what the assignment specifies; WebSocket adds
  streaming and a real "session ended" signal.

## Known limitations
- Not production-hardened: no auth, rate limiting, or PII controls; any caller can read any `user_id`'s memories.
- Per-user write ordering is in-process (an `asyncio.Lock`), so it assumes a single app instance.
- Past-session recall depends on sessions being ended (explicitly, on disconnect, or by the idle sweeper).
- Small models are weak at nuanced extraction and Hindi; the rules/validation layers exist to compensate.
- vLLM's advantage is concurrent throughput (continuous batching, paged KV cache). For a single-user demo it is not
  automatically faster than Ollama/llama.cpp, and it needs a CUDA GPU (or the vllm-metal plugin on a Mac).
- The vLLM client is tested against a fake HTTP server (request shape, SSE streaming, JSON-schema requests, errors),
  not against a live vLLM instance.
- `docker-compose.yml` uses `:latest` for Chroma and vLLM; pin release tags (Chroma's must match the `chromadb` client you install).
- The vLLM compose service has not been run on a GPU yet; if the model fails to load, check `docker compose logs vllm`
  (typical causes: missing `HF_TOKEN` / un-accepted Llama license, or `VLLM_GPU_MEMORY_UTILIZATION` too high for the card).

## Layout
```
app/
  main.py  config.py  container.py  models.py  schemas.py
  api/        http.py  ws.py
  services/   chat_service  retriever  extractor  memory_manager  summarizer  prompt
  memory/     schema (slot vocabulary)  rules (gate + regex extraction)  categories (query keywords)
  stores/     redis_session_store  mongo_memory_store  vector_store  inmemory  base
  llm/        base  openai_compat (vLLM)  ollama  mock     embeddings/  base (Granite + fake)
scripts/     calibrate_similarity (threshold tuning for the real embedder)
cli.py  docker-compose.yml  tests/
```
