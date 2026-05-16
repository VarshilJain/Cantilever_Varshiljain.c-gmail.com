from typing import Any

from app.agent import run_matching_agent
from app.clients import get_llm_client
from app.config import get_settings
from app.models import CandidateSummary, RankedJob, RecommendResponse, RefineResponse
from app.ranking import rank_jobs
from app.tools import generate_clarifying_question


def recommend(resume_text: str) -> RecommendResponse:
    ranked = rank_jobs(resume_text)
    candidate, explained = run_matching_agent(resume_text, ranked)
    question = generate_clarifying_question(resume_text, candidate, ranked)

    explanation_by_id = {int(j["id"]): str(j["explanation"]) for j in explained}
    ranked_jobs = [
        RankedJob(
            id=int(job["id"]),
            title=str(job["title"]),
            company=str(job["company"]),
            similarity_score=float(job.get("similarity_score", 0)),
            explanation=explanation_by_id.get(
                int(job["id"]),
                "Ranked by semantic similarity; detailed reasoning available for top matches.",
            ),
        )
        for job in ranked
    ]

    return RecommendResponse(
        candidate=CandidateSummary(
            name=candidate.get("name", "Unknown"),
            skills=candidate.get("skills", []),
            experience_years=float(candidate.get("experience_years", 0)),
        ),
        ranked_jobs=ranked_jobs,
        clarifying_question=question,
    )


def refine(
    resume_text: str,
    clarifying_question: str,
    candidate_answer: str,
) -> RefineResponse:
    context = (
        f"Clarifying question: {clarifying_question}\n"
        f"Candidate answer: {candidate_answer}"
    )
    reranked = rank_jobs(resume_text, context_suffix=context)
    candidate, explained = run_matching_agent(
        f"{resume_text}\n\n{context}",
        reranked,
    )

    reasoning = _build_refine_reasoning(
        clarifying_question,
        candidate_answer,
        candidate,
        explained,
    )

    explanation_by_id = {int(j["id"]): str(j["explanation"]) for j in explained}
    ranked_jobs = [
        RankedJob(
            id=int(job["id"]),
            title=str(job["title"]),
            company=str(job["company"]),
            similarity_score=float(job.get("similarity_score", 0)),
            explanation=explanation_by_id.get(
                int(job["id"]),
                "Re-ranked by semantic similarity with your clarifying answer.",
            ),
        )
        for job in reranked
    ]

    return RefineResponse(ranked_jobs=ranked_jobs, reasoning=reasoning)


def _build_refine_reasoning(
    question: str,
    answer: str,
    candidate: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> str:
    settings = get_settings()
    client = get_llm_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Explain in 2-4 sentences how the candidate's answer to the clarifying "
                    "question should change job ranking priorities. Be specific."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Answer: {answer}\n"
                    f"Candidate: {candidate}\n"
                    f"New top matches: {jobs}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()
