from langchain.tools import tool
import datetime


@tool("current_time", return_direct=False)
def current_time() -> str:
    """Return current UTC time ISO string."""
    return str(datetime.datetime.utcnow().isoformat())


@tool("word_count", return_direct=False)
def word_count(text: str) -> str:
    """Return number of words in the provided text."""
    n = len([w for w in text.split() if w.strip()])
    return str(n)


TOOLS = [current_time, word_count]
