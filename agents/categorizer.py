from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm


def build_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert analyst that assigns scientific and technical documents to rich topical categories. "
                "Given the provided content, identify the most relevant primary domain, list notable subdomains, "
                "enumerate specific technical topics or methods, and mention potential applications or audiences. "
                "Respond in valid JSON with the keys: primary_domain (string), subdomains (list of strings), "
                "key_topics (list of strings), applications (list of strings), and notes (string).",
            ),
            ("human", "{input}"),
        ]
    )
    return prompt | get_llm() | StrOutputParser()
