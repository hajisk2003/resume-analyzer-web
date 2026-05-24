import fitz
from fastapi import HTTPException


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    if pdf_document.page_count == 0:
        raise HTTPException(status_code=400, detail="PDF has no pages")

    pages = []
    for page_num in range(pdf_document.page_count):
        text = pdf_document[page_num].get_text("text")
        if text.strip():
            pages.append(text)

    pdf_document.close()

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text. PDF may be a scanned image.",
        )

    return full_text


def validate_pdf_file(filename: str, file_size_bytes: int) -> None:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    if file_size_bytes > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")
