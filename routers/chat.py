from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from agents.registry import load_agent, list_agents

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    user_prompt: str = Field(
        description="Это поле с промптом пользователя",
        min_length=5,
        max_length=200,
        examples=["What time is it in UTC?"],
    )
    agent: str = Field(
        default="chat-basic", examples=["chat-basic", "chat-react", "summarizer"]
    )


class ChatResponse(BaseModel):
    agent: str = Field(
        default="chat-basic", examples=["chat-basic", "chat-react", "summarizer"]
    )
    response: str = Field(description="Какой-то ответ")


@router.get("/agents")
def available_agents():
    return {"agents": list_agents()}


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        agent = load_agent(req.agent)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Унифицированный вызов: поддержка Runnable и AgentExecutor
    try:
        if hasattr(agent, "ainvoke"):
            result = await agent.ainvoke({"input": req.user_prompt})  # Runnable
        else:
            # AgentExecutor (sync) — завернем в thread pool на проде; в учебном варианте допускаем sync
            result = agent.invoke({"input": req.user_prompt})
        response = (
            result if isinstance(result, str) else (result.get("output") or str(result))
        )
        return {"agent": req.agent, "response": response}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Agent error: {ex}")
