from typing import List
from langchain.tools import Tool

def find_tool(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"{tool_name}을 가진 Tool을 찾을 수 없습니다.")