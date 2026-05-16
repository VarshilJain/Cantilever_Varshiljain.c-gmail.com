# Technical Write-up — Smart Job Match Agent

---

## 1. Design Choices

### LLM Provider

**Groq** (`llama-3.3-70b-versatile` via the OpenAI-compatible endpoint `https://api.groq.com/openai/v1`) handles tool calling, structured parsing, and clarifying-question generation. Groq's inference speed and cost profile make it well-suited for the low-latency, multi-turn agentic loop this system runs per request.

### Embeddings

Groq does not expose an embeddings API, so the embedding layer is decoupled by environment:

| Environment | Provider | Model | Rationale |
|---|---|---|---|
| Local / dev | `fastembed` | `BAAI/bge-small-en-v1.5` | No second API key; runs in-process |
| Vercel (prod) | OpenAI (`EMBEDDING_PROVIDER=openai`) | `text-embedding-3-small` | Avoids loading a ~130 MB model on cold start within Vercel's memory and 60 s timeout constraints |

### Alternatives Considered

- **xAI Grok** — single vendor simplicity, but a distinct product from Groq with less favorable latency for this use case.
- **`sentence-transformers` only** — too heavy for serverless cold starts.
- **Pure BM25** — fast but produces poor cross-domain match quality; no semantic understanding of skill synonyms or role equivalences.

### Key Trade-offs

- **Single LLM vendor (Groq)** for all inference; optional second key (OpenAI) for production embeddings. Keep this in mind when estimating operational dependencies.
- Vectors use **cosine similarity on L2-normalised embeddings**. Job text concatenates title, skills, domain, and description to produce richer, multi-faceted vectors rather than matching on title alone.

---

## 2. Agentic Architecture

### Request Flow

```
Resume text + top-5 semantic matches
            │
            ▼
    LLM Orchestrator  (tool_choice=auto)
            │
            ├──► Tool 1: parse_resume(resume_text)
            │           → { name, skills, experience_years,
            │               preferred_roles, education }
            │
            └──► Tool 2: explain_matches(candidate, top_jobs)
                        → per-job 2–3 sentence fit explanation
            │
            ▼
    Separate LLM call → clarifying_question
                        (dynamic; not templated)
```

### Why Two Tools Instead of One Monolithic Prompt

Splitting the work across two tools reflects a deliberate separation of concerns:

- **`parse_resume`** is a pure extraction task — low temperature, strict JSON schema, deterministic output. It benefits from being isolated so failures (e.g. hallucinated skills, malformed JSON) are immediately attributable and retryable without re-running the explanation step.
- **`explain_matches`** is a reasoning task — it consumes the structured candidate profile and job context to generate natural-language fit summaries. Temperature and prompt framing differ from extraction.

A single monolithic prompt mixes these two modes, making failures harder to debug and preventing clean re-runs of just the broken step. The orchestrator loop executes *real* `tool_calls` from the OpenAI-compatible API and feeds results back as `role: tool` messages — satisfying the native function-calling requirement rather than simulating it via prompt engineering.

### Why a Separate LLM Call for the Clarifying Question

The clarifying question uses a third LLM call rather than a fourth tool. This keeps the tool contract narrow and predictable: the two registered tools handle extraction and explanation; open-ended question generation does not need a schema or structured output. The trade-off is one additional round-trip, which is acceptable given that clarifying questions are not on the critical latency path.

### Failure Modes

| Mode | Description | Mitigation |
|---|---|---|
| Tool call skipped | Orchestrator skips a tool and responds directly | Fall back to direct tool invocation; bounded iteration cap prevents infinite loops |
| Hallucinated skills | `parse_resume` invents skills not in the resume | Downstream `explain_matches` may overstate fit; partially mitigated by low-temperature schema-constrained extraction |
| Vercel timeout | 3–4 LLM round-trips + embedding batch must complete in < 60 s | Precomputing job embeddings (see §4) is the primary fix |
| Malformed tool arguments | JSON in tool call arguments is invalid | `tool_choice=auto` reduces incidence; error handling wraps each parse |

---

## 3. Known Weaknesses

### Resume Quality Sensitivity

The system accepts raw text only — no PDF pipeline. OCR errors, tables, or bullet-less prose degrade both the embedding vector (noisy text → poor semantic match) and the structured parse (incomplete or incorrect field extraction). Investing in a robust document ingestion layer (e.g. `pdfplumber`, layout-aware extraction) would materially improve match quality before any model changes.

### Scalability

The current architecture is not designed for high concurrency:

- **Embedding**: all jobs are embedded per request — O(n) in catalog size, batched once but still linear. At 10 k concurrent users with a growing job catalog, this becomes the dominant cost and latency driver.
- **No vector store**: similarity search is in-memory. Moving beyond ~50 jobs requires a proper ANN index (Pinecone, Upstash, pgvector).
- **No caching**: identical resumes trigger full re-embedding and re-inference.
- **No async job queues**: each `/recommend` request blocks synchronously on 3–4 LLM calls.

### Scope Cuts (Intentional)

- `jobs.json` is a placeholder; the full dataset has not been integrated.
- `/refine` re-embeds with the clarifying answer appended as context rather than using a learned reranker — effective enough for the prototype but not robust to adversarial or ambiguous answers.
- No authentication or rate limiting on the API.
- No observability: no distributed tracing, latency metrics, or error rate dashboards.

---

## 4. Next Steps

### Priority 1 — Precompute Job Embeddings *(~2 days, highest impact)*

Embed all jobs once at startup (or build time) and cache the vectors in memory or Vercel Edge Config. This eliminates the per-request embedding of the job catalog, reducing `/recommend` latency and cost by an estimated **40–60%**, and is the main bottleneck before catalog growth makes the current approach untenable.

### Priority 2 — Vector Store Integration *(when catalog exceeds ~50 jobs)*

Migrate to Pinecone, Upstash Vector, or pgvector for ANN search. Unlocks sub-millisecond similarity queries at scale and removes the in-memory O(n) search.

### Priority 3 — PDF / Document Ingestion Pipeline

Add layout-aware PDF parsing (e.g. `pdfplumber` or a dedicated document extraction service) to handle real-world resumes submitted as files rather than pasted text.

### Priority 4 — Observability

Add request tracing (e.g. Langfuse or Helicone) to surface per-step latencies, tool call success rates, and embedding quality metrics. Essential before any production rollout.
