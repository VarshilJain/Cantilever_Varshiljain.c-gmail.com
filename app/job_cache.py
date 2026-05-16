from typing import Any

import numpy as np

from app.embeddings import embed_texts
from app.jobs import job_document, load_jobs

_job_matrix: np.ndarray | None = None
_jobs: list[dict[str, Any]] | None = None


def warmup_job_embeddings() -> None:
    global _job_matrix, _jobs
    _jobs = load_jobs()
    texts = [job_document(job) for job in _jobs]
    _job_matrix = embed_texts(texts)


def get_job_embeddings() -> tuple[list[dict[str, Any]], np.ndarray]:
    if _job_matrix is None or _jobs is None:
        warmup_job_embeddings()
    assert _jobs is not None and _job_matrix is not None
    return _jobs, _job_matrix
