from typing import Any

import numpy as np

from app.config import get_settings
from app.embeddings import cosine_similarity, embed_texts
from app.job_cache import get_job_embeddings


def rank_jobs(
    resume_text: str,
    *,
    top_n: int | None = None,
    context_suffix: str = "",
) -> list[dict[str, Any]]:
    settings = get_settings()
    top_n = top_n or settings.top_n_results
    jobs, job_matrix = get_job_embeddings()

    resume_input = resume_text.strip()
    if context_suffix:
        resume_input = f"{resume_input}\n\nAdditional context: {context_suffix.strip()}"

    resume_vec = embed_texts([resume_input])[0]
    scores = cosine_similarity(resume_vec, job_matrix)

    ranked_indices = np.argsort(scores)[::-1][:top_n]
    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        job = jobs[int(idx)]
        results.append(
            {
                **job,
                "similarity_score": round(float(scores[idx]), 4),
            }
        )
    return results
