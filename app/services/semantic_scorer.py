import time
import requests
import numpy as np
from fastapi import HTTPException
from app.core.config import settings

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
MAX_CHARS = 800
MAX_RETRIES = 3
RETRY_DELAY = 5


def get_embedding(text: str) -> list[float] | None:
    headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True},
    }
    try:
        response = requests.post(
            HF_API_URL, headers=headers, json=payload, timeout=15
        )
        if response.status_code == 200:
            embedding = response.json()
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            return embedding
        return None
    except Exception:
        return None


def get_document_embedding(text: str) -> list[float] | None:
    chunks = chunk_text(text)
    if len(chunks) == 1:
        return get_embedding(chunks[0])
    embeddings = [get_embedding(c) for c in chunks[:4]]
    embeddings = [e for e in embeddings if e is not None]
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0).tolist()


def chunk_text(text: str, chunk_size: int = MAX_CHARS) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - 200
    for i in range(0, len(text), step):
        chunk = text[i: i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


def get_document_embedding(text: str) -> list[float]:
    chunks = chunk_text(text)

    if len(chunks) == 1:
        return get_embedding(chunks[0])

    embeddings = [get_embedding(chunk) for chunk in chunks[:4]]
    return np.mean(embeddings, axis=0).tolist()


def cosine_similarity_vectors(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def calculate_semantic_score(resume_text: str, job_description: str) -> dict:
    resume_embedding = get_document_embedding(resume_text)
    jd_embedding = get_document_embedding(job_description)
    similarity = cosine_similarity_vectors(resume_embedding, jd_embedding)
    return {
        "semantic_score": round(similarity * 100, 2),
        "model_used": HF_MODEL,
    }
