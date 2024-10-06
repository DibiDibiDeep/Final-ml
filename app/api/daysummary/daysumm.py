from typing import List
import logging
from uuid import uuid4
from typing import Dict
import json

from langchain.agents.format_scratchpad.log import format_log_to_str

from langchain.schema import AgentAction, AgentFinish

from .tools import (
    retriever_assistant,
    cls_intent_assistant,
    sharing_assistant,
    write_diary_assistant,
    save_diary_assistant,
    except_situation_assistant,
)
from .utils.agent_util import find_tool, setup_agent
from .models import Query
from fastapi import APIRouter, HTTPException


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 사용할 도구들 정의
tools = [
    retriever_assistant,
    cls_intent_assistant,
    sharing_assistant,
    write_diary_assistant,
    save_diary_assistant,
    except_situation_assistant,
]
# 에이전트 설정
agent = setup_agent(tools)

router = APIRouter()

# 사용자 세션 관리를 위한 딕셔너리
user_sessions: Dict[str, List[str]] = {}


@router.post("/process_query")
async def process_user_query(query: Query):
    # 로깅: 쿼리 처리 시작
    logger.info(f"Processing query: {query}")

    # 세션 ID가 없으면 새로 생성
    if not query.session_id:
        session_id = str(uuid4())

    # 쿼리에서 필요한 정보 추출
    input = query.text
    user_id = query.user_id
    baby_id = query.baby_id
    session_id = query.session_id

    # 채팅 기록 가져오기 또는 새로 생성
    chat_history = user_sessions.get(session_id, [])
    chat_history.append(f"User: {input}")
    logger.info(f"Current chat history for session {session_id}: {chat_history}")

    # 출력 형식 설정
    output_format = {"user_id": user_id, "baby_id": baby_id, "session_id": session_id}

    # 에이전트 실행
    intermediate_steps = []
    agent_step = None

    while not isinstance(agent_step, AgentFinish):
        # 에이전트 호출
        agent_step = agent.invoke(
            {
                "input": f"user_id: {user_id}, baby_id: {baby_id}, input: {input}",
                "agent_scratchpad": format_log_to_str(intermediate_steps),
                "chat_history": str(chat_history),
            }
        )

        if isinstance(agent_step, AgentAction):
            # 도구 이름, 입력값 추출 및 도구 실행.
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input
            tool_to_use = find_tool(tools, tool_name)
            observation = tool_to_use.invoke(tool_input)
            logger.info(f"=== Tool Response!!=== \nTool Response: {observation}")

            # retriever_assistant가 결과를 찾지 못한 경우 처리
            if tool_name == "retriever_assistant" and observation == "No results found":
                except_tool = find_tool(tools, "except_situation_assistant")
                except_input = json.dumps(
                    {
                        "query": input,
                        "thought": "retriever_assistant found no results, using except_situation_assistant",
                    }
                )
                observation = except_tool.invoke(except_input)
                agent_step = AgentFinish(
                    return_values={"output": observation},
                    log="Used except_situation_assistant due to no results from retriever_assistant.",
                )

            # 특정 도구들의 경우 출력한 답변을 최종 답변으로 바로 사용.
            if tool_name in [
                "except_situation_assistant",
                "sharing_assistant",
                "write_diary_assistant",
                "save_diary_assistant",
            ]:
                agent_step = AgentFinish(
                    return_values={"output": observation},
                    log=f"{tool_name} tool returned {observation}, ending the agent execution.",
                )

            intermediate_steps.append((agent_step, str(observation)))

        if isinstance(agent_step, AgentFinish) or agent_step:
            result = agent_step

    # 결과 처리
    if output_format.get("response") is None:
        print("=== Agent Finish!!===")
        logger.info(f"=== Agent Finish!!=== \nAgent Response: {result}")
        output_format.update({"response": result.return_values["output"]})

    # 채팅 기록 저장
    chat_history.append(f"Bot: {output_format['response']}")
    user_sessions[session_id] = chat_history
    return output_format

    # 예외 처리
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
