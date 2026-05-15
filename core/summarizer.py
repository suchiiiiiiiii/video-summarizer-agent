from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    # Step 1 — summarize each chunk independently
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a summarization expert. Summarize the following transcript excerpt "
                "concisely. The content may be from a YouTube video, podcast, tutorial, "
                "product review, or meeting — adapt your summary style accordingly. "
                "Preserve key facts, names, products, and opinions mentioned.",
            ),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    # Step 2 — combine into a final structured summary
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert content summarizer. Combine the following partial summaries "
                "into one final, well-structured summary.\n\n"
                "Use this format:\n"
                "- Use ## for main section headings\n"
                "- Use **bold** for product names, key terms, and important points\n"
                "- Use bullet points (- ) for lists of items\n"
                "- Keep the tone informative and clear\n"
                "- Do NOT add a preamble like 'Here is the summary' — start directly\n\n"
                "The content may be a YouTube video, tutorial, review, or meeting. "
                "Adapt the section headings to match the content type naturally.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the transcript excerpt, generate a short, punchy title "
                    "(maximum 8 words). The content may be a YouTube video, podcast, "
                    "tutorial, review, or meeting. Return ONLY the title — no quotes, "
                    "no punctuation at the end, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])