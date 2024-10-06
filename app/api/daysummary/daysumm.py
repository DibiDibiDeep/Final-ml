from typing import List
import logging
from uuid import uuid4
from typing import Dict
import json

from langchain.agents.format_scratchpad.log import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
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
output_parser = ReActSingleInputOutputParser()
router = APIRouter()

# 사용자 세션 관리를 위한 딕셔너리
user_sessions: Dict[str, Dict[str, List[str]]] = {}


# 채팅 기록 초기화 로직 추가.(reset_history)
@router.post("/process_query")
async def process_user_query(query: Query, reset_history: bool = False):
    # 로깅: 쿼리 처리 시작
    logger.info(f"Processing query: {query}")

    # 쿼리에서 필요한 정보 추출
    input = query.text
    user_id = query.user_id
    baby_id = query.baby_id
    # 세션 ID가 없으면 새로 생성
    if not query.session_id:
        session_id = str(uuid4())
    else:
        session_id = query.session_id

    # 사용자 세션 키 생성 (user_id와 session_id 결합)
    user_session_key = f"{user_id}:{session_id}"

    # user_sessions 딕셔너리 초기화 확인
    if user_session_key not in user_sessions:
        user_sessions[user_session_key] = {}

    # 채팅 기록 초기화 로직
    if reset_history:
        user_sessions[user_session_key][baby_id] = []
        logger.info(f"Chat history reset for user {user_id}, baby {baby_id}")
        chat_history = []
    else:
        chat_history = user_sessions[user_session_key].get(baby_id, [])

    chat_history.append(f"User: {input}")

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
        # 에이전트 출력 형식 바람직항 형태로 생성하지 않은 경우 에러 핸들링 로직 추가.
        # 출력 결과 'Action' or 'Final Answer' 가 없는 경우 처리. -> except_situation_assistant 도구 사용하여 답변 생성 후 로직 종료.
        if (
            "Action" not in agent_step.content
            or "Final Answer" not in agent_step.content
        ):
            tool_name = "except_situation_assistant"
            tool_input = json.dumps(
                {
                    "query": agent_step.content.strip(),
                    "thought": "I see that your input is a bit unclear, and I'm not sure how to proceed. Would you like to share more about your day or perhaps ask a specific question? Let me know how I can assist you!",
                }
            )
            tool_to_use = find_tool(tools, tool_name)
            observation = tool_to_use.invoke(tool_input)
            logger.info(f"=== Tool Response!!=== \nTool Response: {observation}")
            agent_step = AgentFinish(
                return_values={"output": observation},
                log="Used except_situation_assistant due to not clear.",
            )
        else:
            # output parser로 파싱이 가능한 경우.
            # 출력 결과 'Action' or 'Final Answer' 가 있는 경우 처리.
            agent_step = output_parser.parse(agent_step.content)
            print(f"agent_step after parse : {agent_step}")

        # 추가 행동이 필요한 경우
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
                    log="Used except_situation_assistant due to not clear.",
                )
            # 사용자 쿼리가 모호하거나 관련이 없는 경우 처리
            if observation == "EXCEPT":
                except_tool = find_tool(tools, "except_situation_assistant")
                except_input = json.dumps(
                    {
                        "query": input,
                        "thought": "I see that your input is a bit unclear, and I'm not sure how to proceed. Would you like to share more about your day or perhaps ask a specific question? Let me know how I can assist you!",
                    }
                )
                observation = except_tool.invoke(except_input)
                agent_step = AgentFinish(
                    return_values={"output": observation},
                    log="Used except_situation_assistant due to difficult to choose the next Action during the Thought process.",
                )

            # 특정 도구들의 경우 출력한 답변을 최종 답변으로 바로 사용. 다음 행동이 필요 없음.
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
    user_sessions[user_session_key][baby_id] = chat_history
    logger.info(
        f"Current chat history for user {user_id}, baby {baby_id}: {chat_history}"
    )

    return output_format

    # 예외 처리
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
