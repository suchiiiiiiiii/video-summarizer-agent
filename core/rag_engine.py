from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriver
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )


def format_doc(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)

    # Bump k to 6 so more context chunks are retrieved
    retriever = get_retriver(vector_store, k=6)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent assistant that answers questions about video/audio content.

You have been given chunks of a transcript below. Answer the user's question 
based on the transcript context. The transcript may be from a YouTube video, 
a podcast, a meeting, a tutorial, a product review, or any spoken content.

Rules:
- Answer directly and concisely using the context
- If the answer is partially available, give what you can and note what's missing
- Only say "I could not find this information" if the topic is completely absent from the context
- If quoting the speaker, say "According to the transcript..."
- Do NOT refuse to answer just because it's not a "meeting" — answer for any content type

Transcript context:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_doc),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriver(vector_store)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent assistant that answers questions about video/audio content.

You have been given chunks of a transcript below. Answer the user's question 
based on the transcript context.

Transcript context:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_doc),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}")
    return answer