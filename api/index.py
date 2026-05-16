from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.job_cache import warmup_job_embeddings
from app.jobs import load_jobs
from app.models import RecommendRequest, RecommendResponse, RefineRequest, RefineResponse
from app.service import recommend, refine

app = FastAPI(
    title="Smart Job Match Agent",
    description="Semantic job ranking with Groq LLM tool-calling agent layer",
    version="1.0.0",
)


@app.on_event("startup")
def warmup() -> None:
    load_jobs()
    settings = get_settings()
    if settings.is_ready():
        try:
            warmup_job_embeddings()
        except Exception:
            pass


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "smart-job-match-agent"}


@app.get("/health")
def health_check() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "jobs_loaded": str(len(load_jobs())),
        "groq_configured": str(settings.is_llm_configured()),
        "embedding_provider": settings.embedding_provider,
        "embedding_ready": str(settings.is_embedding_configured()),
        "llm_model": settings.llm_model,
    }


def _require_ready() -> None:
    settings = get_settings()
    if not settings.is_llm_configured():
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server",
        )
    if not settings.is_embedding_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "Embeddings not configured. Set EMBEDDING_PROVIDER=fastembed (default) "
                "or EMBEDDING_PROVIDER=openai with OPENAI_API_KEY"
            ),
        )


@app.post("/recommend", response_model=RecommendResponse)
def recommend_jobs(payload: RecommendRequest) -> RecommendResponse:
    _require_ready()
    try:
        return recommend(payload.resume_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {exc}",
        ) from exc


@app.post("/refine", response_model=RefineResponse)
def refine_jobs(payload: RefineRequest) -> RefineResponse:
    _require_ready()
    try:
        return refine(
            payload.resume_text,
            payload.clarifying_question,
            payload.candidate_answer,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Refine failed: {exc}",
        ) from exc


@app.exception_handler(ValidationError)
async def validation_exception_handler(_request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
