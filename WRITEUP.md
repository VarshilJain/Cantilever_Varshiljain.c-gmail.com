# Technical Write-up — Smart Job Match Agent

## 1. Design Choices

**LLM provider:** Groq via the OpenAI-compatible API (`https://api.groq.com/openai/v1`), model `llama-3.3-70b-versatile` for tool calling, parsing, and clarifying questions. Groq is fast and cost-effective for the agentic layer.

**Embeddings:** Groq does not expose an embeddings API. Locally, `fastembed` with `BAAI/bge-small-en-v1.5` runs without a second API key. On Vercel, `EMBEDDING_PROVIDER=openai` with `text-embedding-3-small` avoids loading a ~130MB model on cold start (memory and timeout constraints).

**Alternatives considered:** xAI Grok (single vendor but different product from Groq), `sentence-transformers` only (heavy on Vercel), pure BM25 (poor cross-domain matching).

**Trade-offs:** Groq-only for LLM; optional second key (OpenAI) for production embeddings on Vercel. Cosine similarity on L2-normalized vectors. Job text concatenates title, skills, domain, and description for richer vectors.

## 2. Agentic Architecture

```
Resume + top-5 semantic matches
        │
        ▼
   LLM (orchestrator, tool_choice=auto)
        │
        ├─► Tool 1: parse_resume(resume_text)
        │         → {name, skills, experience_years, preferred_roles, education}
        │
        └─► Tool 2: explain_matches(candidate, top_jobs)
                  → per-job 2–3 sentence fit explanations
        │
        ▼
   Separate LLM call → clarifying_question (dynamic, not templated)
```

**Why two tools instead of one prompt?** Separation of concerns: parsing is extraction (low temperature, JSON schema); explanation is reasoning over structured candidate + job context. A monolithic prompt mixes tasks, makes failures harder to debug, and does not satisfy the requirement for *native* function/tool calling. The orchestrator loop executes real `tool_calls` from the OpenAI API and feeds results back as `role: tool` messages.

**Failure modes:** (1) Model skips a tool call—we fall back to direct tool invocation. (2) Hallucinated skills in `parse_resume`—downstream explanations may overstate fit. (3) Timeout on Vercel (60s)—multiple LLM round-trips plus embedding batch must complete in one request. (4) Tool argument JSON malformed—mitigated by `tool_choice=auto` and bounded iteration cap.

## 3. Honest Weaknesses

**Noisy resumes:** OCR errors, tables, or bullet-less prose degrade both embeddings and structured parsing. No PDF pipeline—only raw text in.

**Scale (10k concurrent):** Synchronous embedding of all jobs per request is O(n) API calls batched once but still linear in catalog size; no vector DB, caching, or precomputed job index. Agent loop adds 3–4 LLM calls per request. Would need precomputed job embeddings in Redis/pgvector and async job queues.

**Corners cut:** Placeholder `jobs.json` until full dataset added; `/refine` re-embeds with answer appended as context rather than a learned reranker; no auth/rate limiting; no observability (tracing/metrics). Clarifying question uses a third LLM call instead of a fourth tool—for speed and simpler contract.

## 4. Next Steps

**Highest-impact improvement (2 days):** Precompute and cache job embeddings at startup (or build time) and store in memory or Edge Config. This cuts latency and cost per `/recommend` by ~40–60% and is the main bottleneck before catalog growth. Second: add a vector store (e.g. Pinecone/Upstash) when moving beyond 50 jobs.

---

*Cold starts on Vercel free tier may add 2–3s on first request; document in demo.*
