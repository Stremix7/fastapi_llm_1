from fastapi import APIRouter, UploadFile, File, HTTPException
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
