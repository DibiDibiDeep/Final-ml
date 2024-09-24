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
from .utils.get_data import get_today_info
from .tools import (
    retreiver_about_qeustion,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")

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
            You have access to the following user information:
            - User ID: {user_id}
            - Baby ID: {baby_id}

            Your task is to determine the user's intent and respond accordingly. You can use the classify_intent_tool to help you, but you should also use your own judgment based on the context of the conversation.

            Possible intents are:
            1. QUESTION: For inquiries about specific events, activities, or details of the parent's or child's day.
            2. DIARY_REQUEST: When the user explicitly requests to write a diary entry.
            3. ANSWER: For cases that are neither QUESTION nor DIARY_REQUEST.

            Based on the intent you determine, follow these steps:

            1. If Intent is 'QUESTION':
               - Use the retreiver_about_qeustion tool to answer the question. Always include the user_id and baby_id when using this tool.
               - If the retreiver_about_qeustion tool returns "No results found", inform the user that there is no information available for their query and explain that the data for that day might not be stored in the database.
               - Questions are connected to previous questions.
               - If the date cannot be determined from the user's query, request information about the year and month.
               - For questions about the child, encourage parents to ask their child directly and suggest follow-up questions.
               - Provide guidance on how to phrase questions to encourage open-ended responses from the child.
               
            2. If Intent is 'DIARY_REQUEST':
               - Do not use any tools.
               - Write a diary entry from the parent's perspective in a casual, informal tone.
               - Base the diary on Today's info and chat history.
               - Focus on the parent's personal thoughts, feelings, and reflections about their day and their child's activities.

            3. If Intent is 'ANSWER':
               - Do not use any tools.
               - Provide a direct answer to the question based on the chat history and Today's info.

            General guidelines:
            - Include a question that can help write a diary entry summarizing the day in all responses except for 'DIARY_REQUEST'.
            - Use tools only for 'QUESTION' intent, and only once per request.
            - Write a diary entry only when the user explicitly requests it ('DIARY_REQUEST' intent).
            - Always respond in Korean.
            """,
        ),
        ("human", "user_id: {user_id}, baby_id: {baby_id}\nToday's info: {today_info}\nQuery: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# LLM and tools setup
llm = ChatOpenAI(
    model_name=llm_model, openai_api_key=openai_api_key, temperature=0
)

tools = [retreiver_about_qeustion]

# Agent setup
agent = (
    {
        "input": lambda x: x["input"],
        "today_info": lambda x: x["today_info"]["today_text"],
        "user_id": lambda x: x["user_id"],
        "baby_id": lambda x: x["baby_id"],
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
    max_iterations=2,
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
        today_info = get_today_info(query.user_id,query.baby_id)
        result = agent_executor.invoke(
            {
                "input": query.text,
                "chat_history": chat_history,
                "today_info": today_info,
                "user_id": query.user_id,
                "baby_id": query.baby_id,
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
