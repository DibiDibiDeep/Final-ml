from typing import List
from langchain.tools import Tool
import os
from typing import Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI


def setup_agent(tools: List[str]):
    # env setting
    llm_model = os.getenv("LLM_MODEL")
    openai_key = os.getenv("OPENAI_API_KEY")

    # template load
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    prompts_dir = os.path.join(
        parent_dir, "prompts", "agent", "daysummary_react_prompt.txt"
    )

    with open(prompts_dir, "r") as f:
        template = f.read()

    # Prompt setting
    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools),
    )

    # llm setting
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        openai_api_key=openai_key,
        stop=["\nObservation", "Observation"],
    )

    # agent setting
    agent: Union[AgentAction, AgentFinish] = (
        # prompt | llm | ReActSingleInputOutputParser()
        prompt
        | llm
    )

    return agent


# Agent가 사용할 도구 이름 찾는 함수
def find_tool(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"{tool_name}을 가진 Tool을 찾을 수 없습니다.")
