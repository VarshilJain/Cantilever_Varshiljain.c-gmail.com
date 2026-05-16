import json
from typing import Any

from app.clients import get_llm_client
from app.config import get_settings
from app.jobs import job_document


def parse_resume(resume_text: str) -> dict[str, Any]:
    """Extract structured candidate fields from raw resume text."""
    client = get_llm_client()
    settings = get_settings()
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured candidate information from the resume. "
                    "Return JSON with keys: name (string), skills (array of strings), "
                    "experience_years (number), preferred_roles (array of strings), "
                    "education (string). Use reasonable defaults when missing."
                ),
            },
            {"role": "user", "content": resume_text},
        ],
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return {
        "name": str(data.get("name", "Unknown")),
        "skills": [str(s) for s in data.get("skills", [])],
        "experience_years": float(data.get("experience_years", 0)),
        "preferred_roles": [str(r) for r in data.get("preferred_roles", [])],
        "education": str(data.get("education", "")),
    }


def explain_matches(
    candidate: dict[str, Any],
    top_jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate 2-3 sentence fit explanations for each top job."""
    client = get_llm_client()
    settings = get_settings()
    jobs_payload = [
        {
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "similarity_score": job.get("similarity_score"),
            "summary": job_document(job),
        }
        for job in top_jobs
    ]
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a career advisor. For each job, write a 2-3 sentence explanation "
                    "of why it is or is not a strong fit for this specific candidate. "
                    "Be concrete and reference skills, experience, domain, and location/remote. "
                    'Return JSON: {"explanations": [{"id": number, "explanation": string}, ...]} '
                    "with one entry per job id provided."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"candidate": candidate, "jobs": jobs_payload},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or '{"explanations": []}'
    data = json.loads(content)
    explanation_map = {
        int(item["id"]): str(item["explanation"])
        for item in data.get("explanations", [])
        if "id" in item and "explanation" in item
    }
    results = []
    for job in top_jobs:
        job_id = int(job["id"])
        results.append(
            {
                "id": job_id,
                "title": job["title"],
                "company": job["company"],
                "similarity_score": job.get("similarity_score", 0.0),
                "explanation": explanation_map.get(
                    job_id,
                    "Fit assessment unavailable for this role.",
                ),
            }
        )
    return results


def generate_clarifying_question(
    resume_text: str,
    candidate: dict[str, Any],
    ranked_jobs: list[dict[str, Any]],
) -> str:
    """Generate one specific follow-up question based on resume and matches."""
    client = get_llm_client()
    settings = get_settings()
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate exactly one smart follow-up question for a job candidate. "
                    "The question must resolve a specific ambiguity or gap you noticed "
                    "when comparing their resume to the top matched jobs. "
                    "Do NOT use generic questions like 'Tell me more about yourself'. "
                    "Do NOT use templates. Reference concrete details (remote work, "
                    "missing skills, domain preference, experience level, etc.). "
                    "Return only the question text, no quotes or preamble."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "resume_excerpt": resume_text[:2000],
                        "candidate": candidate,
                        "top_matches": [
                            {
                                "id": j["id"],
                                "title": j["title"],
                                "company": j["company"],
                                "domain": j.get("domain"),
                                "remote": j.get("remote"),
                                "skills": j.get("skills"),
                                "experience_years": j.get("experience_years"),
                            }
                            for j in ranked_jobs[:5]
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()
