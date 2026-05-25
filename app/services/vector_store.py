import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.data.sample_jobs import SAMPLE_JOBS

logger = logging.getLogger(__name__)

_vectorizer = None
_job_matrix = None
_jobs_metadata = None


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_index() -> None:
    global _vectorizer, _job_matrix, _jobs_metadata

    job_texts = []
    metadata = []

    for job in SAMPLE_JOBS:
        text = preprocess(f"{job['title']} {job['description']}")
        job_texts.append(text)
        metadata.append({
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "url": job["url"],
        })

    _vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    _job_matrix = _vectorizer.fit_transform(job_texts)
    _jobs_metadata = metadata

    logger.info(f"Local job index built: {len(metadata)} jobs")


def load_index() -> None:
    build_index()


def get_similar_jobs(resume_text: str, top_k: int = 3) -> list[dict]:
    global _vectorizer, _job_matrix, _jobs_metadata

    if _vectorizer is None:
        build_index()

    processed = preprocess(resume_text)
    query_vector = _vectorizer.transform([processed])
    scores = cosine_similarity(query_vector, _job_matrix)[0]

    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        job = _jobs_metadata[idx].copy()
        job["similarity_score"] = round(float(scores[idx]) * 100, 2)
        results.append(job)

    return results

# import json
# import logging
# from pathlib import Path

# import faiss
# import numpy as np

# from app.data.sample_jobs import SAMPLE_JOBS
# from app.services.semantic_scorer import get_embedding

# logger = logging.getLogger(__name__)

# DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# INDEX_PATH = DATA_DIR / "jobs.index"
# JOBS_META_PATH = DATA_DIR / "jobs_meta.json"

# _index = None
# _jobs_metadata = None


# def build_index() -> None:
#     global _index, _jobs_metadata

#     logger.info("Building FAISS index from job descriptions...")
#     embeddings = []
#     metadata = []

#     for job in SAMPLE_JOBS:
#         text = f"{job['title']} {job['description']}"
#         emb = get_embedding(text)
#         embeddings.append(emb)
#         metadata.append({
#             "id": job["id"],
#             "title": job["title"],
#             "company": job["company"],
#             "location": job["location"],
#             "url": job["url"],
#         })

#     vectors = np.array(embeddings, dtype=np.float32)
#     dimension = vectors.shape[1]

#     faiss.normalize_L2(vectors)
#     _index = faiss.IndexFlatIP(dimension)
#     _index.add(vectors)

#     DATA_DIR.mkdir(parents=True, exist_ok=True)
#     faiss.write_index(_index, str(INDEX_PATH))
#     INDEX_PATH.with_suffix(".json")
#     with open(JOBS_META_PATH, "w") as f:
#         json.dump(metadata, f)

#     _jobs_metadata = metadata
#     logger.info(f"FAISS index built: {_index.ntotal} jobs indexed")


# def load_index() -> None:
#     global _index, _jobs_metadata

#     if INDEX_PATH.exists() and JOBS_META_PATH.exists():
#         _index = faiss.read_index(str(INDEX_PATH))
#         with open(JOBS_META_PATH) as f:
#             _jobs_metadata = json.load(f)
#         logger.info(f"FAISS index loaded: {_index.ntotal} jobs")
#     else:
#         build_index()


# def get_similar_jobs(resume_text: str, top_k: int = 3) -> list[dict]:
#     global _index, _jobs_metadata

#     if _index is None:
#         load_index()

#     resume_emb = get_embedding(resume_text[:800])
#     query_vector = np.array([resume_emb], dtype=np.float32)
#     faiss.normalize_L2(query_vector)

#     D, I = _index.search(query_vector, top_k)

#     results = []
#     for score, idx in zip(D[0], I[0]):
#         if idx == -1:
#             continue
#         job = _jobs_metadata[idx].copy()
#         job["similarity_score"] = round(float(score) * 100, 2)
#         results.append(job)

#     return results
