import time
import requests
from fastapi import HTTPException
from app.core.config import settings

HF_LLM_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
MAX_RETRIES = 3
RETRY_DELAY = 5


def call_hf_llm(prompt: str, max_new_tokens: int = 300) -> str:
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.3,
        },
        "options": {"wait_for_model": True, "use_cache": False},
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(HF_LLM_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
                raise HTTPException(status_code=503, detail="Unexpected LLM response format")

            if response.status_code in (503, 429):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue

            raise HTTPException(
                status_code=503,
                detail=f"LLM service unavailable: {response.status_code}",
            )

        except HTTPException:
            raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise HTTPException(status_code=503, detail="LLM service timed out")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM service error: {str(e)}")

    raise HTTPException(status_code=503, detail="LLM service failed after retries")


def build_suggestion_prompt(
    resume_text: str,
    job_description: str,
    matched_keywords: list,
    missing_keywords: list,
    final_score: float,
) -> str:
    missing_str = ", ".join(missing_keywords[:10]) if missing_keywords else "none"
    matched_str = ", ".join(matched_keywords[:10]) if matched_keywords else "none"

    return f"""You are a professional resume coach. Analyze this resume against the job description and provide exactly 5 specific improvement suggestions.

Job Description:
{job_description[:500]}

Resume:
{resume_text[:800]}

Match Score: {final_score}/100
Keywords found: {matched_str}
Keywords missing: {missing_str}

Provide exactly 5 specific, actionable suggestions:
1.
2.
3.
4.
5."""


def parse_suggestions(raw_text: str) -> list[dict]:
    suggestions = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            clean = line.lstrip("0123456789.-) ").strip()
            if len(clean) > 10:
                suggestions.append({
                    "category": categorize_suggestion(clean),
                    "suggestion": clean,
                    "priority": "high" if len(suggestions) < 2 else "medium",
                })

    if not suggestions:
        suggestions.append({
            "category": "General",
            "suggestion": raw_text[:500],
            "priority": "medium",
        })

    return suggestions[:5]


def categorize_suggestion(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["skill", "technology", "tool", "framework", "language"]):
        return "Skills"
    if any(w in text_lower for w in ["experience", "project", "achievement", "accomplish"]):
        return "Experience"
    if any(w in text_lower for w in ["keyword", "ats", "term", "phrase"]):
        return "Keywords"
    if any(w in text_lower for w in ["format", "structure", "layout", "section"]):
        return "Format"
    return "Content"


def generate_resume_suggestions(
    resume_text: str,
    job_description: str,
    matched_keywords: list,
    missing_keywords: list,
    final_score: float,
) -> list[dict]:
    prompt = build_suggestion_prompt(
        resume_text, job_description, matched_keywords, missing_keywords, final_score
    )
    raw_output = call_hf_llm(prompt)
    return parse_suggestions(raw_output)
