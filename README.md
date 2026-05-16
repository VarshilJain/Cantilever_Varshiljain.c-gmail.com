# Smart Job Match Agent

Semantic job ranking (embeddings + cosine similarity) with a **Groq** LLM agent (native tool calling) and FastAPI, deployable on Vercel.

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your **Groq** key from [console.groq.com](https://console.groq.com/):

```
GROQ_API_KEY=gsk_your_key_here
```

Run:

```bash
python main.py
```

Test:

```bash
curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\"resume_text\": \"Python developer with NLP and FastAPI experience.\"}"
```

Docs: http://127.0.0.1:8000/docs

## API keys

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Yes | LLM — agent, parsing, explanations, clarifying question |
| `OPENAI_API_KEY` | Only if `EMBEDDING_PROVIDER=openai` | Embeddings on Vercel (recommended for deploy) |
| (none extra) | Default | `EMBEDDING_PROVIDER=fastembed` uses local model locally |

**Groq does not offer embeddings.** Default `fastembed` needs no second key for local dev. For **Vercel**, set `EMBEDDING_PROVIDER=openai` and add `OPENAI_API_KEY` (embeddings only) to avoid large cold-start model downloads.

## Deploy to Vercel

From the project root (`Cantilever_Labs`):

```bash
npm i -g vercel
vercel login
vercel
```

Follow prompts to **create/link** a project. Then production deploy:

```bash
vercel --prod
```

### Vercel environment variables

**Project → Settings → Environment Variables** (Production + Preview):

| Name | Value |
|------|--------|
| `GROQ_API_KEY` | your Groq key |
| `EMBEDDING_PROVIDER` | `openai` (recommended on Vercel) |
| `OPENAI_API_KEY` | OpenAI key (embeddings only) |

Redeploy after saving env vars.

### Project not showing on Vercel?

The project only appears after you run `vercel` once or import the GitHub repo. A local `python main.py` does not create a Vercel project.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status and config check |
| POST | `/recommend` | Rank jobs + agent + clarifying question |
| POST | `/refine` | Re-rank after clarifying answer |
