from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function

import logging
from .config import vector_store
from .utils.get_data import today_info
from .tools import (
    parent_retriever_assistant,
    child_retriever_assistant,
    classify_intent,
)
from .utils.preprocess import add_sample_data

router = APIRouter()

# User session management
user_sessions: Dict[str, List[str]] = {}


class Query(BaseModel):
    text: str
    session_id: str = None


class DiaryEntry(BaseModel):
    name: str
    emotion: str
    health: str
    nutrition: str
    activities: List[str]
    social: str
    special: str
    keywords: List[str]
    diary: str


# Agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant designed to help with questions about a person's day and write diary entries.
            Based on the provided intent classification, follow these steps:
            1. If Intent is 'QUESTION_CHILD', use child_retriever_assistant tool filtering by role: child.
            2. If Intent is 'QUESTION_PARENT', use parent_retriever_assistant tool filtering by role: parents.
            3. If Intent is 'DIARY_REQUEST':
               a. Write a diary entry from the perspective of the parent, addressing themselves in a casual, based on the Today's info and chat history.
                  When writing the diary, focus on the parent's personal thoughts, feelings, and reflections about their day and their child's activities.
            4. If Intent is 'OTHER':
               a. Generate a follow-up question that helps the parent reflect on their emotions and memories of the day.
               b. Formulate a question that offers comfort and encouragement, focusing on the parent's personal experiences and feelings about their day and their child's activities.
            Use each tool only once per request.
            """,
        ),
        ("human", "Intent: {intent}\nToday's info: {today_info}\nQuery: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# LLM and tools setup
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
tools = [parent_retriever_assistant, child_retriever_assistant]

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Agent setup
agent = (
    {
        "input": lambda x: x["input"],
        "intent": lambda x: x["intent"],
        "today_info": lambda x: x["today_info"]["today_text"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    | OpenAIFunctionsAgentOutputParser()
)

# AgentExecutor setup
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=4,
    return_intermediate_steps=True,
    early_stopping_method="force",
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/process_query")
async def process_user_query(query: Query):
    try:
        if not query.session_id:
            query.session_id = str(uuid4())

        chat_history = user_sessions.get(query.session_id, [])
        chat_history.append(f"User: {query.text}")

        logger.info(
            f"Current chat history for session {query.session_id}: {chat_history}"
        )
        intent_result = classify_intent.invoke(query.text)
        result = agent_executor.invoke(
            {
                "input": query.text,
                "intent": intent_result,
                "chat_history": chat_history,
                "today_info": today_info,
            }
        )

        chat_history.append(f"Assistant: {result['output']}")

        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        user_sessions[query.session_id] = chat_history

        logger.info(
            f"Updated chat history for session {query.session_id}: {chat_history}"
        )

        return {"response": result["output"], "session_id": query.session_id}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_diary")
async def add_diary(diary: DiaryEntry):
    try:
        document_content = f"날짜: {diary.date}\n이름: {diary.name}\n감정: {diary.emotion}\n건강: {diary.health}\n영양: {diary.nutrition}\n활동: {', '.join(diary.activities)}\n사회적 활동: {diary.social}\n특별한 일: {diary.special}\n키워드: {', '.join(diary.keywords)}"
        vector_store.add_documents(documents=[document_content], ids=[str(uuid4())])
        return {"message": "Diary entry added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.on_event("startup")
async def startup_event():
    add_sample_data(vector_store)
