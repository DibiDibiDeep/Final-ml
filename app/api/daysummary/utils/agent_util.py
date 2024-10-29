from typing import List
from langchain.tools import Tool
import os
from typing import Union
import re
import json

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

def valid_output_format(text: str) -> bool:
    """
    Checks if the output format is valid for ReActSingleInputOutputParser.
    
    Returns:
    - True if the format is valid
    - False if the format is invalid
    """
    # Check for Final Answer format
    if "Final Answer:" in text:
        return True
    
    # Check for Action and Action Input format
    action_input_regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    if re.search(action_input_regex, text, re.DOTALL):
        return True
    
    # If neither format is found, it's invalid
    return False

async def handle_exception(input: str, thought: str, tools: List) -> AgentFinish:
    except_tool = find_tool(tools, "except_situation_assistant")
    except_input = json.dumps({"query": input, "thought": thought})
    observation = await except_tool.ainvoke(except_input)
    return AgentFinish(
        return_values={"output": observation},
        log=f"Used except_situation_assistant due to exception.",
    )