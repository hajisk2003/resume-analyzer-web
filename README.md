# Resume Analyzer

AI-powered resume analysis and job matching system built with FastAPI.

## What It Does

Upload a resume PDF and paste a job description. The system returns:
- TF-IDF keyword match score
- Semantic similarity score using BERT embeddings
- Combined final match score (0–100)
- Matched and missing keywords
- 5 LLM-generated improvement suggestions
- Top 3 similar jobs from the job database

## Tech Stack

- **Backend** — FastAPI, SQLAlchemy, PostgreSQL, Alembic
- **ML** — scikit-learn (TF-IDF), HuggingFace Inference API (embeddings + LLM)
- **Vector Search** — FAISS
- **Auth** — JWT (python-jose + bcrypt)
- **PDF** — PyMuPDF
- **Frontend** — HTML/CSS/JS (single file)
- **Deploy** — Render (backend) + Neon.tech (PostgreSQL)

## Local Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env           # fill in your values

alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the API docs.  
Open `frontend/index.html` in your browser for the UI.

## Environment Variables

| Variable | Description |
|----------|-------------|
| DATABASE_URL | PostgreSQL connection string |
| SECRET_KEY | Random string for JWT signing |
| HUGGINGFACE_API_KEY | Free token from huggingface.co |
| DEBUG | True for local, False for production |

## Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /auth/register | Create account |
| POST | /auth/login | Get JWT token |
| GET | /auth/me | Current user |
| POST | /resume/upload | Upload PDF + job description |
| POST | /resume/{id}/full-analysis | Run complete analysis |
| GET | /resume/{id}/similar-jobs | Find matching jobs |
| GET | /resume/my-resumes | List your resumes |
