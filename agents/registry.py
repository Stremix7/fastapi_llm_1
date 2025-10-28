from typing import Any, Dict, Callable

from . import summarizer, categorizer

_REGISTRY: Dict[str, Callable[[], Any]] = {
    "summarizer": summarizer.build_agent,
    "categorizer": categorizer.build_agent,
}


def list_agents():
    return sorted(_REGISTRY.keys())


def load_agent(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list_agents()}")
    return _REGISTRY[name]()
