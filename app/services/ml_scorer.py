import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    cleaned = clean_text(text)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    matrix = vectorizer.fit_transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [word for word, score in ranked[:top_n] if score > 0]


def calculate_tfidf_score(resume_text: str, job_description: str) -> dict:
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    score = round(float(similarity) * 100, 2)

    jd_keywords = set(extract_keywords(job_description, top_n=20))
    resume_keywords = set(extract_keywords(resume_text, top_n=30))

    return {
        "tfidf_score": score,
        "matched_keywords": list(jd_keywords & resume_keywords),
        "missing_keywords": list(jd_keywords - resume_keywords),
    }


def calculate_final_score(tfidf_score: float, semantic_score: float) -> float:
    return round((tfidf_score * 0.4) + (semantic_score * 0.6), 2)
