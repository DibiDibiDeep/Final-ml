import logging
import time

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
from .utils.agent_util import (
     find_tool, 
     setup_agent, 
     valid_output_format, 
     handle_exception
     )
from .models import Query, AIChatHistory
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
# 출력 파서 설정
output_parser = ReActSingleInputOutputParser()

# 사용자 채팅 기록 관리 클래스 정의
chat_history_manager = AIChatHistory()

router = APIRouter()

# 채팅 기록 초기화 로직 추가.(reset_history)
@router.post("/process_query")
async def process_user_query(query: Query, reset_history: bool = False):
    start_time = time.time()  # 시작 시간 기록
    # 로깅: 쿼리 처리 시작
    logger.info(f"Processing query: {query}")

    # 쿼리에서 필요한 정보 추출
    input = query.text
    user_id = query.user_id
    baby_id = query.baby_id
    # 세션 ID가 없으면 새로 생성
    session_id = chat_history_manager.get_or_create_session(user_id, query.session_id)
    # 출력 형식 설정
    output_format = {"user_id": user_id, "baby_id": baby_id, "session_id": session_id}

    # 채팅 기록 초기화 로직
    if reset_history:
        chat_history_manager.reset_history(user_id, session_id, baby_id)
        logger.info(f"Chat history reset for user {user_id}, baby {baby_id}")

    # 사용자 입력 채팅 기록에 추가
    chat_history_manager.add_message(user_id, session_id, baby_id, input)

    # 에이전트 실행
    intermediate_steps = []
    agent_step = None

    while not isinstance(agent_step, AgentFinish):
        # 에이전트 호출
        chat_history = chat_history_manager.get_full_history(user_id, session_id, baby_id)
        agent_step = agent.invoke(
            {
                "input": f"user_id: {user_id}, baby_id: {baby_id}, input: {input}",
                "agent_scratchpad": format_log_to_str(intermediate_steps),
                "chat_history": str(chat_history),
            }
        )
        if not valid_output_format(agent_step.content):
                agent_step = await handle_exception(
                    input,
                    "I see that your input is a bit unclear, and I'm not sure how to proceed. Would you like to share more about your day or perhaps ask a specific question? Let me know how I can assist you!",
                    tools
                )
        else:
            # output parser로 파싱이 가능한 경우.
            # 출력 결과 'Action' or 'Final Answer' 가 있는 경우 처리.
            agent_step = output_parser.parse(agent_step.content)

        # 추가 행동이 필요한 경우
        if isinstance(agent_step, AgentAction):
            # 도구 이름, 입력값 추출 및 도구 실행.
            tool_name = agent_step.tool
            tool_input = agent_step.tool_input
            tool_to_use = find_tool(tools, tool_name)
            observation = tool_to_use.invoke(tool_input)
            logger.info(f"=== Tool Response!!=== \nTool Response: {observation}")

            # RAG 결과 없을 경우 except_situation_assistant 사용
            if tool_name == "retriever_assistant" and observation == 'No results found':
                    agent_step = await handle_exception(
                        input,
                        "retriever_assistant found no results, using except_situation_assistant",
                        tools
                    )
            # 사용자 쿼리가 모호하거나 관련이 없는 경우 처리
            elif observation == 'EXCEPT':
                agent_step = await handle_exception(
                    input,
                    "I see that your input is a bit unclear, and I'm not sure how to proceed. Would you like to share more about your day or perhaps ask a specific question? Let me know how I can assist you!",
                    tools
                )
            # 특정 도구들의 경우 출력한 답변을 최종 답변으로 바로 사용. 다음 행동이 필요 없음.
            elif tool_name in ['except_situation_assistant', "sharing_assistant", "write_diary_assistant"]:
                agent_step = AgentFinish(
                    return_values={"output": observation},
                    log=f"{tool_name} tool returned {observation}, ending the agent execution.",
                )
            # 중간 단계 저장
            intermediate_steps.append((agent_step, str(observation)))
        # 최종 답변인 경우
        if isinstance(agent_step, AgentFinish) or agent_step:
            result = agent_step

    # 결과 처리
    if output_format.get("response") is None:
        print("=== Agent Finish!!===")
        logger.info(f"=== Agent Finish!!=== \nAgent Response: {result}")
        output_format.update({"response": result.return_values["output"]})

    # 챗봇 답변 저장
    chat_history_manager.add_message(user_id, session_id, baby_id, output_format['response'], is_user=False)
    logger.info(
        f"Current chat history for user {user_id}, baby {baby_id}: {chat_history_manager.get_full_history(user_id, session_id, baby_id)}"
    )
    end_time = time.time()  # 종료 시간 기록
    processing_time = end_time - start_time
    logger.info(f"Query processing time: {processing_time:.2f} seconds")
    return output_format

    # 예외 처리
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
