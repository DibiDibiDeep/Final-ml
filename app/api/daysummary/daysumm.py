from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from uuid import uuid4
import os
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import convert_to_openai_function

from .tools import (
    retreiver_about_qeustion
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

current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(current_dir, "prompts", "daysummary_prompt_ver2.txt")

with open(
    prompt_path, "r", encoding="utf-8"
) as file:
    template = file.read()

# Agent prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        ("human", "user_id: {user_id}, baby_id: {baby_id}\nQuery: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# LLM and tools setup
llm = ChatOpenAI(
    model_name=llm_model, openai_api_key=openai_api_key, temperature=0
)


tools = [
    retreiver_about_qeustion
    ]

# Agent setup
agent = (
    {
        "input": lambda x: x["input"],
        "user_id": lambda x: x["user_id"],
        "baby_id": lambda x: x["baby_id"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm.bind(functions=[convert_to_openai_function(t) for t in tools])
    | OpenAIFunctionsAgentOutputParser()
)

# AgentExecutor setup
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process_query")
async def process_user_query(query: Query):
    try:
        logger.info(f"Processing query: {query}")
        if not query.session_id:
            query.session_id = str(uuid4())

        chat_history = user_sessions.get(query.session_id, [])
        chat_history.append(f"User: {query.text}")
        logger.info(
            f"Current chat history for session {query.session_id}: {chat_history}"
        )
        result = agent_executor.invoke(
            {
                "input": query.text,
                "chat_history": chat_history,
                "user_id": query.user_id,
                "baby_id": query.baby_id,
            }
        )

        chat_history.append(f"Assistant: {result['output']}")


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
