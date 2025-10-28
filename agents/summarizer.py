from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import get_llm


def build_agent():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the content in 3-4 bullet points, be precise."),
            ("human", "{input}"),
        ]
    )
    return prompt | get_llm() | StrOutputParser()
