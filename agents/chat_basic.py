from langchain_core.runnables import Runnable
from core.llm import simple_chain


def build_agent() -> Runnable:
    return simple_chain("You are a concise assistant. Reply shortly.")
