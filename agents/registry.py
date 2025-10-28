from typing import Any, Dict, Callable

from . import chat_basic, chat_react, summarizer

_REGISTRY: Dict[str, Callable[[], Any]] = {
    "chat-basic": chat_basic.build_agent,
    "chat-react": chat_react.build_agent,
    "summarizer": summarizer.build_agent,
}


def list_agents():
    return sorted(_REGISTRY.keys())


def load_agent(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list_agents()}")
    return _REGISTRY[name]()
