import json
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from core.llm import get_llm
from .tools import TOOLS

# --- Формат подсказки для ReAct ---
# Модель должна выдавать либо:
#  1) промежуточный шаг:
#     Thought: ...
#     Action: <tool_name>
#     Action Input: <input>
#  2) финальный ответ:
#     Thought: ...
#     Final Answer: <answer>
SYSTEM_PROMPT = """You are a helpful assistant with tool-usage. You can reason step by step.
You have access to the following tools:
{tool_manifest}

When you decide to use a tool, follow EXACTLY this format:

Thought: <your reasoning>
Action: <one of [{tool_names}]>
Action Input: <a short input for the selected tool>

If you have enough information to answer the user, produce:

Thought: <your reasoning>
Final Answer: <your final, concise answer>

Be concise and avoid unnecessary text.
"""

HUMAN_TEMPLATE = """Question: {input}

Scratchpad (previous steps):
{scratchpad}
"""


def _tools_index():
    """Подготовим словарь инструментов по имени."""
    return {t.name: t for t in TOOLS}


def _build_prompt():
    tool_manifest_lines = []
    for t in TOOLS:
        desc = (t.description or "").strip()
        tool_manifest_lines.append(f"- {t.name}: {desc}")
    tool_manifest = "\n".join(tool_manifest_lines)
    tool_names = ", ".join([t.name for t in TOOLS])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT.format(
                    tool_manifest=tool_manifest, tool_names=tool_names
                ),
            ),
            ("human", HUMAN_TEMPLATE),
        ]
    )
    return prompt


def _parse_step(text: str) -> Dict[str, Any]:
    """
    Выделяем блоки вида:
      Thought: ...
      Action: TOOL
      Action Input: INPUT
    или финальный ответ:
      Thought: ...
      Final Answer: ANSWER
    Возвращаем словарь с ключами: {type: "action"/"final", ...}
    """
    # Сначала — финальный ответ
    m_final = re.search(r"Final Answer:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m_final:
        answer = m_final.group(1).strip()
        return {"type": "final", "answer": answer}

    # Иначе — действие
    m_action = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", text)
    m_input = re.search(r"Action Input:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

    if m_action and m_input:
        tool = m_action.group(1).strip()
        action_input = m_input.group(1).strip()
        # Часто LLM кладёт JSON в Action Input — попробуем распарсить, но мягко.
        try:
            action_input_json = json.loads(action_input)
            action_input_val = action_input_json
        except Exception:
            action_input_val = action_input
        return {"type": "action", "tool": tool, "tool_input": action_input_val}

    # Если ничего не распознали — трактуем как «непонятно», пусть агент завершит
    return {"type": "final", "answer": text.strip()}


async def _react_call(inputs: Dict[str, Any]) -> str:
    """
    Асинхронная ReAct-петля:
    - генерируем шаг (Thought/Action/Final Answer),
    - при Action исполняем инструмент, добавляем Observation,
    - продолжаем до Final Answer или достижения лимита итераций.
    """
    question: str = inputs.get("input", "")
    max_iters: int = int(inputs.get("max_iters", 5))

    llm = get_llm()
    prompt = _build_prompt()
    chain = prompt | llm | StrOutputParser()

    tools_by_name = _tools_index()
    scratchpad: List[str] = []

    for _ in range(max_iters):
        rendered = await chain.ainvoke(
            {
                "input": question,
                "scratchpad": "\n".join(scratchpad) if scratchpad else "(empty)",
            }
        )

        step = _parse_step(rendered)

        if step["type"] == "final":
            # Возвращаем финальный ответ модели
            return step["answer"]

        if step["type"] == "action":
            tool_name = step["tool"]
            tool_input = step["tool_input"]

            if tool_name not in tools_by_name:
                scratchpad.append(
                    f"Thought: Selected tool '{tool_name}' is not available.\n"
                    f"Observation: Tool '{tool_name}' not found. Available: {list(tools_by_name.keys())}"
                )
                # Дадим модели шанс скорректироваться на следующем шаге
                continue

            tool = tools_by_name[tool_name]
            try:
                # Инструменты из langchain.tools по умолчанию синхронны
                observation = tool.invoke(tool_input)
            except Exception as e:
                observation = f"ToolError: {e}"
                print(observation)

            scratchpad.append(
                f"Thought: Executing tool.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {tool_input}\n"
                f"Observation: {observation}"
            )
            continue

    # Если достигли лимита итераций — возвращаем краткий вывод из последнего шага
    return "Reached max iterations without a final answer."


def build_agent():
    """
    Возвращаем Runnable с поддержкой invoke/ainvoke.
    Совместимо с твоими роутерами: можно вызывать agent.ainvoke({"input": ...})
    """
    return RunnableLambda(_react_call)
