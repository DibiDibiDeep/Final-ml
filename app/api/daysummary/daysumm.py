from typing import List, Union
from dotenv import load_dotenv
import os
from pydantic import BaseModel
import logging
from uuid import uuid4
from typing import Dict

from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser, ReActOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI

from .tools import (
    retriever_assistant, 
    cls_intent_assistant,
    sharing_assistant,
    write_diary_assistant,
    save_diary_assistant
    )
from .utils.agent_util import find_tool

from fastapi import APIRouter, HTTPException

llm_model = os.getenv("LLM_MODEL")
openai_key = os.getenv("OPENAI_API_KEY")
# template load
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_dir = os.path.join(current_dir, "prompts/agent", 'daysummary_react_prompt.txt')
with open(prompts_dir, "r") as f:
    template = f.read()
tools = [
    retriever_assistant, 
    cls_intent_assistant,
    sharing_assistant,
    write_diary_assistant,
    save_diary_assistant
    ]

# Prompt setting
prompt = PromptTemplate.from_template(template)
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join(t.name for t in tools),
)

llm = ChatOpenAI(
        model = llm_model,
        temperature=0,
        openai_api_key=openai_key,
        stop=["\nObservation", "Observation"],
        )
# agent = prompt | llm | ReActSingleInputOutputParser()
agent: Union[AgentAction, AgentFinish] = prompt | llm | ReActOutputParser()

class Query(BaseModel):
    baby_id: int = 1
    user_id: int = 1
    session_id: str = None
    text: str

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# User session management
user_sessions: Dict[str, List[str]] = {}

@router.post("/process_query")
async def process_user_query(query: Query):
    # try:
    logger.info(f"Processing query: {query}")
    if not query.session_id:
        session_id = str(uuid4())

    input = query.text
    user_id = query.user_id
    baby_id = query.baby_id
    session_id = query.session_id

    chat_history = user_sessions.get(session_id, [])
    chat_history.append(f"User: {input}")
    logger.info(
        f"Current chat history for session {session_id}: {chat_history}"
    )
    output_format = {"user_id": user_id, "baby_id": baby_id, "session_id": session_id}
    
    # Invoke
    intermediate_steps = []
    agent_step = None

    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {
                "input": f"user_id: {user_id}, baby_id: {baby_id}, input: {input}",
                "agent_scratchpad": format_log_to_str(intermediate_steps),
                "chat_history": str(chat_history),
            }
        )

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input
            tool_to_use = find_tool(tools, tool_name)
            observation = tool_to_use.invoke(tool_input)
            print(f"{observation=}")


            intermediate_steps.append((agent_step, str(observation)))
                

        if isinstance(agent_step, AgentFinish) or agent_step:
            result = agent_step

    if output_format.get("response") is None:
        print("=== Agent Finish!!===")
        print(f"{result=}")
        output_format.update({"response": result.return_values['output']})
    
    # Save chat history
    chat_history.append(f"Bot: {output_format['response']}")
    user_sessions[session_id] = chat_history
    return output_format
                
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))