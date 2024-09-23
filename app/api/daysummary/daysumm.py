from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from uuid import uuid4
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function

import logging

# from .config import collection
from .utils.get_data import today_info
from .tools import (
    retreiver_about_qeustion,
    classify_intent,
)

# User session management
user_sessions: Dict[str, List[str]] = {}


class Query(BaseModel):
    baby_id: int
    user_id: int
    session_id: str = None
    text: str


# Agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant designed to help with questions about a person's day and write diary entries.
            Your answers should always include a question that can help write a diary entry summarizing the day, except when the intent is 'DIARY_REQUEST'.
            Based on the provided intent classification, follow these steps:
            1. If Intent is 'QUESTION', use retreiver_about_qeustion tool and answer the question.
               - Question is connected to before question.
               - If the date cannot be determined from the user's query, request the user to provide the information about the Year, Month.
               - When the question is about the child, encourage the parent to ask their child directly about their day, activities, or feelings.
               - Suggest follow-up questions that the parent can ask their child to gain more insight into the child's experiences.
               - Provide guidance on how to phrase questions in a way that encourages open-ended responses from the child.
            2. If Intent is 'DIARY_REQUEST':
               a. No use tool and write a diary entry from the perspective of the parent, addressing themselves in a casual informal tone.
               b. based on the Today's info and chat history(question and answer).chat history may contain questions about past events.
               c. When writing the diary, focus on the parent's personal thoughts, feelings, and reflections about their day and their child's activities.
            3. If Intent is 'ANSWER':
               a. No use tool and answer to the question. And another question that can help write a diary entry summarizing the day.
             Use each tool only once per request. And always in Korean.
            """,
        ),
        ("human", "Intent: {intent}\nToday's info: {today_info}\nQuery: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# LLM and tools setup
llm = ChatOpenAI(
    model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0
)
# tools = [parent_retriever_assistant, child_retriever_assistant]
tools = [retreiver_about_qeustion]

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

router = APIRouter()


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

        return {
            "baby_id": query.baby_id,
            "user_id": query.user_id,
            "session_id": query.session_id,
            "response": result["output"],
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
