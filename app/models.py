from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)

    @field_validator("resume_text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("resume_text cannot be empty or whitespace only")
        return stripped


class RefineRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    clarifying_question: str = Field(..., min_length=1)
    candidate_answer: str = Field(..., min_length=1)

    @field_validator("resume_text", "clarifying_question", "candidate_answer")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace only")
        return stripped


class CandidateSummary(BaseModel):
    name: str
    skills: list[str]
    experience_years: float


class RankedJob(BaseModel):
    id: int
    title: str
    company: str
    similarity_score: float
    explanation: str


class RecommendResponse(BaseModel):
    candidate: CandidateSummary
    ranked_jobs: list[RankedJob]
    clarifying_question: str


class RefineResponse(BaseModel):
    ranked_jobs: list[RankedJob]
    reasoning: str
