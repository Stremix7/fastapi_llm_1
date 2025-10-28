from fastapi import FastAPI, Request
from routers import chat, files, health
import time

app = FastAPI(title="LLM Agent API")


@app.middleware("http")
async def latency_header(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time"] = f"{time.time() - start:.3f}"
    return resp


app.include_router(chat.router)
app.include_router(files.router)
app.include_router(health.router)


@app.get("/")
def root():
    return {"message": "FastAPI + LangChain demo is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
