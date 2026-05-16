import json
from typing import Any

from app.clients import get_llm_client
from app.config import get_settings
from app.tools import explain_matches, parse_resume

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "parse_resume",
            "description": (
                "Extract structured candidate fields from raw resume text: "
                "name, skills, experience_years, preferred_roles, education."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_text": {
                        "type": "string",
                        "description": "Full raw resume text provided by the candidate.",
                    }
                },
                "required": ["resume_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_matches",
            "description": (
                "Generate natural-language fit explanations for the top semantic job matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate": {
                        "type": "object",
                        "description": "Structured candidate profile from parse_resume.",
                    },
                    "top_jobs": {
                        "type": "array",
                        "description": "Top 5 jobs from semantic similarity ranking.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["candidate", "top_jobs"],
            },
        },
    },
]


def _execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "parse_resume":
        return parse_resume(arguments["resume_text"])
    if name == "explain_matches":
        return explain_matches(arguments["candidate"], arguments["top_jobs"])
    raise ValueError(f"Unknown tool: {name}")


def run_matching_agent(
    resume_text: str,
    top_jobs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Two-step agentic flow using native LLM tool calling:
    1. parse_resume
    2. explain_matches (on top-5 semantic matches)
    """
    settings = get_settings()
    client = get_llm_client()
    top_five = top_jobs[: settings.top_n_for_agent]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a job-matching agent. You MUST complete exactly two tool calls in order:\n"
                "1. Call parse_resume with the candidate's resume text.\n"
                "2. Call explain_matches using the parsed candidate and the provided top_jobs.\n"
                "Do not answer without calling both tools."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Resume text:\n{resume_text}\n\n"
                f"Top semantic matches (use these for explain_matches):\n"
                f"{json.dumps(top_five, ensure_ascii=False)}"
            ),
        },
    ]

    parsed_candidate: dict[str, Any] | None = None
    explained_jobs: list[dict[str, Any]] | None = None
    max_iterations = 6

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            break

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments or "{}")
            result = _execute_tool(fn_name, fn_args)

            if fn_name == "parse_resume":
                parsed_candidate = result
            elif fn_name == "explain_matches":
                explained_jobs = result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    if parsed_candidate is None:
        parsed_candidate = parse_resume(resume_text)
    if explained_jobs is None:
        explained_jobs = explain_matches(parsed_candidate, top_five)

    return parsed_candidate, explained_jobs
