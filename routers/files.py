from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException
from PyPDF2 import PdfReader

from agents.registry import load_agent

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload-and-summarize")
async def upload_and_summarize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".txt", ".md", ".json")):
        raise HTTPException(status_code=400, detail="Allowed: .txt, .md, .json")

    content = (await file.read()).decode("utf-8", errors="ignore")
    agent = load_agent("summarizer")

    if hasattr(agent, "ainvoke"):
        summary = await agent.ainvoke({"input": content})
    else:
        summary = agent.invoke({"input": content})

    return {"filename": file.filename, "summary": summary}


@router.post("/upload-and-categorize")
async def upload_and_categorize(file: UploadFile = File(...)):
    allowed_extensions = (".txt", ".md", ".json", ".pdf")
    filename = file.filename.lower()

    if not filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, detail="Allowed: .txt, .md, .json, .pdf"
        )

    if filename.endswith(".pdf"):
        raw_bytes = await file.read()
        try:
            pdf = PdfReader(BytesIO(raw_bytes))
        except Exception as exc:  # pragma: no cover - depends on PDF content
            raise HTTPException(status_code=400, detail="Failed to read PDF") from exc

        pages_text = [
            (page.extract_text() or "").strip()
            for page in pdf.pages
        ]
        content = "\n\n".join(filter(None, pages_text))

        if not content.strip():
            raise HTTPException(
                status_code=400, detail="PDF does not contain extractable text"
            )
    else:
        content = (await file.read()).decode("utf-8", errors="ignore")

    agent = load_agent("categorizer")

    if hasattr(agent, "ainvoke"):
        categorization = await agent.ainvoke({"input": content})
    else:
        categorization = agent.invoke({"input": content})

    return {"filename": file.filename, "categorization": categorization}
