import json
from functools import lru_cache
from pathlib import Path
from typing import Any

JOBS_PATH = Path(__file__).resolve().parent.parent / "jobs.json"


@lru_cache(maxsize=1)
def load_jobs() -> list[dict[str, Any]]:
    if not JOBS_PATH.exists():
        raise FileNotFoundError(f"jobs dataset not found at {JOBS_PATH}")
    with JOBS_PATH.open(encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list) or len(jobs) == 0:
        raise ValueError("jobs.json must be a non-empty array")
    return jobs


def job_document(job: dict[str, Any]) -> str:
    skills = ", ".join(job.get("skills", []))
    remote = "remote" if job.get("remote") else "on-site"
    return (
        f"Title: {job.get('title', '')}. "
        f"Company: {job.get('company', '')}. "
        f"Location: {job.get('location', '')} ({remote}). "
        f"Domain: {job.get('domain', '')}. "
        f"Experience required: {job.get('experience_years', '')} years. "
        f"Skills: {skills}. "
        f"Description: {job.get('description', '')}"
    )
