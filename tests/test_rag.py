import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import build_index, get_similar_jobs

RESUME = """
Python developer with 3 years experience.
Skills: FastAPI, PostgreSQL, Docker, REST APIs, SQLAlchemy, JWT authentication.
Built and deployed microservices on AWS. Familiar with machine learning pipelines.
"""

if __name__ == "__main__":
    print("Building FAISS index...")
    build_index()

    print("\nSearching for similar jobs...")
    jobs = get_similar_jobs(RESUME, top_k=3)

    for job in jobs:
        print(f"\n{job['title']} at {job['company']}")
        print(f"Location: {job['location']}")
        print(f"Similarity: {job['similarity_score']}%")
