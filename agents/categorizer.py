from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm


def build_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — эксперт-аналитик, распределяющий научные и технические документы по обширным тематическим категориям. На основе предоставленного контента определи наиболее релевантный основной домен, перечисли значимые поддомены, перечисли конкретные технические темы или методы, а также укажи потенциальные области применения или аудитории, где они могут быть использованы. Предоставь корректный JSON-файл с ключами: primary_domain (строка), subdomains (список строк), key_topics (список строк), applications (список строк) и notes (строка)."
            ),
            ("human", "{input}"),
        ]
    )
    return prompt | get_llm() | StrOutputParser()
