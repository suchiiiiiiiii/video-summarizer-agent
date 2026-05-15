# Actionable items, decisions, questions
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Analyze the following transcript carefully — "
        "it may be a meeting, a YouTube video, a tutorial, a review, or any spoken content.\n\n"
        "Extract ALL action items, tasks, recommendations, or things the speaker says "
        "the audience/participants should DO. Include product recommendations, tips, "
        "steps to follow, or any call-to-action.\n\n"
        "Format your response as a numbered list ONLY, one item per line.\n"
        "Example:\n"
        "1. Try double cleansing before your skincare routine\n"
        "2. Apply sunscreen as the last step in your morning routine\n\n"
        "If truly none exist, return exactly: NONE"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Analyze the following transcript carefully — "
        "it may be a meeting, a YouTube video, a tutorial, a review, or any spoken content.\n\n"
        "Extract ALL key decisions, conclusions, verdicts, rankings, winners, "
        "or definitive statements made. Include product verdicts, comparisons resolved, "
        "opinions stated as facts, or any clear conclusion reached.\n\n"
        "Format your response as a numbered list ONLY, one item per line.\n"
        "Example:\n"
        "1. King Anoa Cleansing Oil ranked #1 in Korea for 7 years\n"
        "2. Double cleansing is essential before foam cleansing\n\n"
        "If truly none exist, return exactly: NONE"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Analyze the following transcript carefully — "
        "it may be a meeting, a YouTube video, a tutorial, a review, or any spoken content.\n\n"
        "Extract ALL open questions, unresolved topics, things mentioned as needing "
        "further research, audience questions raised, or topics left for follow-up.\n\n"
        "Format your response as a numbered list ONLY, one item per line.\n"
        "Example:\n"
        "1. Which cleansing oil works best for sensitive skin?\n"
        "2. Is fragrance-free always better for all skin types?\n\n"
        "If truly none exist, return exactly: NONE"
    )
    return chain.invoke(transcript)