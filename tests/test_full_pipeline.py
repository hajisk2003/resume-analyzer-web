import time
import requests

BASE = "http://localhost:8000"
PDF_PATH = "tests/sample_resume.pdf"

JOB_DESCRIPTION = """
We are looking for a senior Python developer with strong experience in FastAPI,
PostgreSQL, Docker, and machine learning. The candidate should know REST API design,
SQLAlchemy, and have deployed ML models. Experience with scikit-learn, numpy, and
pandas is required. Knowledge of JWT authentication and cloud deployment is a plus.
"""


def register_and_login():
    requests.post(f"{BASE}/auth/register", json={
        "email": "pipeline@example.com",
        "password": "SecurePass123",
        "full_name": "Pipeline Tester"
    })
    res = requests.post(f"{BASE}/auth/login", data={
        "username": "pipeline@example.com",
        "password": "SecurePass123"
    })
    token = res.json().get("access_token")
    print(f"Token: {token[:30]}...")
    return token


def upload_resume(token):
    with open(PDF_PATH, "rb") as f:
        res = requests.post(
            f"{BASE}/resume/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("resume.pdf", f, "application/pdf")},
            data={"job_description": JOB_DESCRIPTION, "job_title": "Senior Python Developer"}
        )
    print(f"Upload: {res.status_code}")
    data = res.json()
    print(f"Resume ID: {data.get('resume_id')} | Text length: {data.get('text_length')}")
    return data.get("resume_id")


def run_full_analysis(token, resume_id):
    print("\nRunning full analysis...")
    start = time.time()
    res = requests.post(
        f"{BASE}/resume/{resume_id}/full-analysis",
        headers={"Authorization": f"Bearer {token}"}
    )
    elapsed = time.time() - start
    print(f"Full analysis: {res.status_code} in {elapsed:.1f}s")

    if res.status_code == 200:
        data = res.json()
        s = data["scores"]
        print(f"TF-IDF: {s['tfidf_score']} | Semantic: {s['semantic_score']} | Final: {s['final_score']}")
        print(f"Matched: {data['keywords']['matched'][:5]}")
        print(f"Missing: {data['keywords']['missing'][:5]}")
        print(f"Suggestions: {len(data['suggestions'])}")
        print(f"Similar jobs: {len(data['similar_jobs'])}")
    else:
        print(f"Error: {res.json()}")


def test_cache(token, resume_id):
    print("\nTesting cache...")
    start = time.time()
    res = requests.post(
        f"{BASE}/resume/{resume_id}/full-analysis",
        headers={"Authorization": f"Bearer {token}"}
    )
    elapsed = time.time() - start
    data = res.json()
    print(f"Cached call: {elapsed:.1f}s | cached={data.get('cached')}")


def test_unauthorized(resume_id):
    res = requests.post(f"{BASE}/resume/{resume_id}/full-analysis")
    print(f"\nNo token → {res.status_code} (expected 401)")


if __name__ == "__main__":
    print("=== Full Pipeline Test ===\n")
    token = register_and_login()
    resume_id = upload_resume(token)
    run_full_analysis(token, resume_id)
    test_cache(token, resume_id)
    test_unauthorized(resume_id)
    print("\n=== Done ===")
