from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from uuid import uuid4
from functools import wraps

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function

from langchain.tools.retriever import create_retriever_tool
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from app.api.daysummary.utils.preprocess import convert_data_structure, create_document_from_data

# Load environment variables
load_dotenv()

# 임베딩 모델 설정
embeddings_model = HuggingFaceEmbeddings(
    # Embedding model 변경 가능
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Milvus 벡터 저장소 설정 (로컬 개발용)
URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": URI},
)

router = APIRouter()

# 사용자 세션을 관리하기 위한 딕셔너리
user_sessions: Dict[str, List[str]] = {}


class Query(BaseModel):
    text: str
    session_id: str = None


# 도구들에 use_once 데코레이터 적용
@tool
def classify_intent(query: str) -> str:
    """
    Classifies the intent of the user's query into more specific categories.
    Use this tool to determine the user's intention and guide the conversation flow.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Classify the user's query into one of these categories:
                'QUESTION_CHILD': for queries about the child's day or activities
                'QUESTION_PARENT': for queries about the parent's day or activities
                'DIARY_REQUEST': for requests to write a diary entry
                'EMOTION_EXPRESSION': for expressions of emotion or mood
                'CLARIFICATION': for requests for more information or clarification
                'OTHER': for anything that doesn't fit the above categories
                
                Provide only the category name as the response.""",
            ),
            ("user", "{query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return response.content.strip().upper()


# 부모 정보 검색을 위한 retriever 생성
parent_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "filter": {"role": "parents"}},  # 부모 역할에 대한 필터 추가
)

# 자녀 정보 검색을 위한 retriever 생성
child_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "filter": {"role": "child"}},  # 자녀 역할에 대한 필터 추가
)


@tool
def parent_rag_assistant(query: str) -> str:
    """
    Retrieves information about the parent's day and activities and generates a response.
    Use this tool when you need to find specific information about the parent's past events.
    """
    return generate_response_from_rag(parent_retriever, query, "parent")


@tool
def child_rag_assistant(query: str) -> str:
    """
    Retrieves information about the child's day and activities and generates a response.
    Use this tool when you need to find specific information about the child's past events.
    """
    return generate_response_from_rag(child_retriever, query, "child")


def generate_response_from_rag(retriever, query: str, role: str) -> str:
    """
    Generates a response based on the retrieved information from RAG.
    """
    # Retrieve relevant information
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate response using LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are an assistant that provides information about a {role}'s day based on the given context. Respond in a natural, conversational tone as if speaking directly to the {role}.",
            ),
            ("human", "Context: {context}"),
            ("human", "Query: {query}"),
            (
                "human",
                f"Generate a response about the {role}'s day based on the context:",
            ),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return response.content.strip()


@tool
def diary_writer(context: str, chat_history: List[str]) -> str:
    """
    Generates a brief diary entry based on the given context and chat history.
    Use this tool when you need to summarize a full day's activities and include user interactions.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that writes brief diary entries. Based on the given context and chat history, write a short diary entry (3-5 sentences) summarizing the key events, information, and user interactions.",
            ),
            ("user", "Context: {context}"),
            ("user", "Chat History: {chat_history}"),
            ("user", "Write a brief diary entry:"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {"context": context, "chat_history": "\n".join(chat_history)}
    )
    return response.content


@tool
def parent_question_generator():
    """
    Generates a follow-up question about the parent's day or activities based on the given context.
    Use this tool to create engaging questions specifically about the parent.
    """
    # 임의(디비에서 현재 날짜 사용자 ID기준으로 데이터 추출 과정 필요)
    today_parent_events = {
        "date": "2024-09-04",
        "role": "parents",
        "emotion": "걱정되지만 희망적이에요",
        "health": "감기 기운이 있어요",
        "nutrition": "따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요.",
        "activities": ["재택근무", "아이 숙제 도와주기", "병원 방문"],
        "social": "화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다.",
        "special": "아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
        "keywords": ["재택근무", "숙제", "병원", "화상회의", "성장"],
        "text": "2024-09-04 걱정되지만 희망적이에요 감기 기운이 있어요 따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요. 재택근무, 아이 숙제 도와주기, 병원 방문 화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다. 아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
    }

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant designed to generate follow-up questions about a parent's day. Based on the given context, create an engaging question that encourages further discussion about the parent's activities, feelings, or experiences.",
            ),
            ("human", "{context}"),
            ("human", "Generate a follow-up question about the parent's day:"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": today_parent_events["text"]})
    return response.content


@tool
def child_question_generator():
    """
    Generates a follow-up question about the child's day or activities based on the given context.
    Use this tool to create engaging questions specifically about the child.
    """
    # 임의(디비에서 현재 날짜 사용자 ID기준으로 데이터 추출 과정 필요)
    today_child_events = {
        "date": "2024-09-04",
        "role": "child",
        "emotion": "궁금하고 걱정돼요",
        "health": "기침이 조금 나요",
        "nutrition": "입맛이 없어서 평소보다 적게 먹었어요.",
        "activities": ["온라인 수업", "책 읽기", "퍼즐 맞추기"],
        "social": "아픈 친구에게 영상통화로 안부를 물었어요.",
        "special": "혼자서 옷 단추를 다 채웠어요! 엄마가 정말 기뻐하셨어요.",
        "keywords": ["온라인수업", "책", "퍼즐", "단추", "영상통화"],
        "text": "오늘은 조금 이상한 하루였어요. 🤒\n학교에 가지 않고 집에서 온라인으로 수업을 들었어요. 선생님 얼굴을 화면으로 보는 게 신기했어요.\n기침이 나고 몸이 안 좋아서 공부하기가 조금 힘들었지만, 엄마가 계속 옆에서 도와주셨어요. 💖\n점심 먹고 나서는 새로 산 공룡 퍼즐을 맞췄어요. 어려웠지만 재미있었어요! 🦕\n저녁에는 혼자서 옷 단추를 다 채웠어요. 엄마가 정말 기뻐하시던게 기억나요. 😊\n아픈 친구한테 전화해서 괜찮은지 물어봤어요. 내일은 우리 둘 다 나아있기를 바라요. 🙏",
    }

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant designed to generate follow-up questions about a child's day. Based on the given context, create an engaging question that encourages further discussion about the child's activities, feelings, or experiences.",
            ),
            ("human", "{context}"),
            ("human", "Generate a follow-up question about the child's day:"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": today_child_events["text"]})
    return response.content


# Agent 프롬프트 수정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant designed to help with questions about a person's day. Follow these steps strictly:

            1. Based on the provided intent classification, use ONLY ONE of the following tools ONCE:
               - For 'QUESTION_CHILD' intent: Use child_rag_assistant
               - For 'QUESTION_PARENT' intent: Use parent_rag_assistant
               - For 'DIARY_REQUEST' intent: Use diary_writer
               - For 'EMOTION_EXPRESSION', 'CLARIFICATION', or 'OTHER' intents: Do not use any additional tools

            2. After using the appropriate tool once, summarize the information directly and concisely.

            3. Provide a final response to the user's query based on the information obtained.

            4. Do not use any tool more than once, and do not use tools that aren't specified for the given intent.

            Remember, your goal is to provide a helpful and concise response while adhering to these guidelines.""",
        ),
        ("human", "Intent: {intent}\nQuery: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

# LLM 및 tools 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
tools = [
    parent_rag_assistant,
    child_rag_assistant,
    diary_writer,
    parent_question_generator,
    child_question_generator,
]

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Agent 설정
agent = (
    {
        "input": lambda x: x["input"],
        "intent": lambda x: x["intent"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    | OpenAIFunctionsAgentOutputParser()
)

# AgentExecutor 설정
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=4,  # 최대 반복 횟수를 3으로 설정
    return_intermediate_steps=True,
    early_stopping_method="force",
)


import logging

# chat history 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/process_query")
async def process_user_query(query: Query):
    try:
        # 세션 ID가 없으면 새로 생성
        if not query.session_id:
            query.session_id = str(uuid4())

        # 세션 ID에 해당하는 채팅 기록 가져오기 (없으면 새로 생성)
        chat_history = user_sessions.get(query.session_id, [])
        chat_history.append(f"User: {query.text}")

        # 현재 채팅 기록 상태 로깅
        logger.info(
            f"Current chat history for session {query.session_id}: {chat_history}"
        )
        intent_result = classify_intent.invoke(query.text)
        result = agent_executor.invoke(
            {"input": query.text, "intent": intent_result, "chat_history": chat_history}
        )

        # 결과 처리 로직
        if (
            not result["output"]
            or "I don't have enough information" in result["output"]
        ):
            result["output"] = (
                "I apologize, but I don't have enough detailed information about your activities today. "
                "Based on the general information available, your day likely included routine activities "
                "such as meals, work or school, and possibly some personal time. If you'd like more specific "
                "information, please provide more context or ask about a particular part of your day."
            )

        chat_history.append(f"Assistant: {result['output']}")

        # 채팅 기록의 길이를 제한하여 메모리 사용을 관리합니다
        if len(chat_history) > 20:  # 예: 최근 20개의 메시지만 유지
            chat_history = chat_history[-20:]

        user_sessions[query.session_id] = chat_history

        # 업데이트된 채팅 기록 상태 로깅
        logger.info(
            f"Updated chat history for session {query.session_id}: {chat_history}"
        )

        return {"response": result["output"], "session_id": query.session_id}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 샘플 일기 데이터 형식 정의
class DiaryEntry(BaseModel):
    name: str
    emotion: str
    health: str
    nutrition: str
    activities: List[str]
    social: str
    special: str
    keywords: List[str]
    diary: str


# 샘플 데이터 추가(삭제 예정)
@router.post("/add_diary")
async def add_diary(diary: DiaryEntry):
    try:
        document_content = f"날짜: {diary.date}\n이름: {diary.name}\n감정: {diary.emotion}\n건강: {diary.health}\n영양: {diary.nutrition}\n활동: {', '.join(diary.activities)}\n사회적 활동: {diary.social}\n특별한 일: {diary.special}\n키워드: {', '.join(diary.keywords)}"
        vector_store.add_documents(documents=[document_content], ids=[str(uuid4())])
        return {"message": "Diary entry added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Sample data addition function
def add_sample_data():
    samples = [
        {
            "date": "2024-09-01",
            "role": "parents",
            "emotion": "뿌듯하고 감사해요",
            "health": "약간 피곤하지만 괜찮아요",
            "nutrition": "아이와 함께 건강한 채소 위주의 식사를 했어요. 브로콜리 스프가 특히 맛있었어요!",
            "activities": ["아이 등하원", "업무", "저녁 책"],
            "social": "동료들과 협력하여 프로젝트를 무사히 마쳤어요. 저녁에는 이웃과 잠깐 대화를 나눴어요.",
            "special": "아이가 어제 배운 노래를 불러주었는데, 정말 감동이었어요. 아이의 성장을 눈으로 확인할 수 있어 행복했습니다.",
            "keywords": ["등하원", "프로젝트", "산책", "노래", "성장"],
            "text": "2024-09-01 뿌듯하고 감사해요 약간 피곤하지만 괜찮아요 아이와 함께 건강한 채소 위주의 식사를 했어요. 브로콜리 스프가 특히 맛있었어요! 아이 등하원, 업무, 저녁 산책 동료들과 협력하여 프로젝트를 무사히 마쳤어요. 저녁에는 이웃과 잠깐 대화를 나눴어요. 아이가 어제 배운 노래를 불러주었는데, 정말 감동이었어요. 아이의 성장을 눈으로 확인할 수 있어 행복했습니다.",
        },
        {
            "date": "2024-09-01",
            "role": "child",
            "emotion": "즐거움과 신남",
            "health": "좋음",
            "nutrition": "식사에 대한 정보는 제공되지 않았습니다.",
            "activities": [
                "아이스크림가게 역할놀이",
                "놀이터에서 놀기",
                "붓으로 그림 그리기",
            ],
            "social": "친구들과 함께 즐겁게 놀며 웃음소리를 나누었습니다.",
            "special": "아이스크림 먹는 연기를 잘 했고, 그림 그리면서 웃음이 끊이지 않았습니다.",
            "keywords": ["역할놀이", "아이스크림", "놀이터", "그림", "웃음"],
            "text": "오늘은 정말 즐거운 하루였어! 😄\n아침에 친구들이랑 아이스크림 가게 역할놀이를 했어. 🍦\n나는 아이스크림을 팔고, 친구들은 손님이 되었지!\n아이스크림 먹는 연기를 정말 잘했어! 😋\n그 다음에는 놀이터에 가서 신나게 놀았어. 🛝\n미끄럼틀도 타고, 그네도 타고, 정말 재밌었어!\n친구들과 함께 웃음소리가 끊이지 않았어. 😂\n마지막으로 붓으로 그림을 그렸는데, 너무 즐거웠어! 🎨\n그림을 그리면서도 계속 웃고 있었어.\n오늘 하루가 너무 행복했어! 💖",
        },
        {
            "date": "2024-09-02",
            "role": "parents",
            "emotion": "조금 지쳤지만 보람차요",
            "health": "허리가 약간 아파요",
            "nutrition": "아침은 오트밀, 점심은 회사 구내식당, 저은 아이와 함께 삼계탕을 먹었어요.",
            "activities": ["아이 학교 준비물 챙기기", "업무 회의", "가족 저녁 식사"],
            "social": "팀 회의에서 새로운 아이디어를 제안했어요. 저녁에는 가족과 오랜만에 대화의 시간을 가졌습니다.",
            "special": "아이가 학교에서 받아온 칭찬스티커를 보여줬는데, 정말 자랑스러웠어요.",
            "keywords": ["준비물", "회의", "삼계탕", "대화", "칭찬스티커"],
            "text": "2024-09-02 조금 지쳤지만 보람차요 허리가 약간 아파요 아침은 오트밀, 점심은 회사 구내식당, 저녁은 아이와 함께 삼계탕을 먹었어요. 아이 학교 준비물 챙기기, 업무 회의, 가족 저녁 식사 팀 회의에서 새로운 아이디어를 제안했어요. 저녁에는 가족과 오랜만에 대화의 시간을 가졌습니다. 아이가 학교에서 받아온 칭찬스티커를 보여줬는데, 정말 자랑스러웠어요.",
        },
        {
            "date": "2024-09-02",
            "role": "child",
            "emotion": "신나고 자신감 넘쳐요",
            "health": "활기차고 건강해요",
            "nutrition": "학교 급식을 맛있게 먹었어요. 특히 디트가 맛있었대요.",
            "activities": ["체육 수업", "미술 시간", "방과후 피아노 레슨"],
            "social": "체육 시간에 친구들과 협동해서 릴레이 경기에서 1등했어요!",
            "special": "미술 시간에 그린 그림을 선생님께서 칭찬해주셨어요.",
            "keywords": ["체육", "릴레이", "미술", "피아노", "칭찬"],
            "text": "오늘은 정말 멋진 하루였어요! 💪\n체육 시간에 친구들이랑 릴레이 경기를 했는데, 우리 팀이 1등을 했어요! 🏃‍♂️🥇\n다 같이 힘을 합쳐서 뛰었더니 정말 뿌듯했어요.\n미술 시간에는 우리 가족 그림을 그렸는데, 선생님께서 정말 잘 그렸대요. 🎨👨‍👩‍👧\n방과 후에는 피아노 레슨도 갔어요. 새로운 곡을 배웠는데 조금 어려웠지만 열심히 연습할 거예요! 🎹\n오늘 하루는 정말 자신감이 넘치는 날이었어요. 내일도 이렇게 잘 할 수 있을 것 같아요! 😊",
        },
        {
            "date": "2024-09-03",
            "role": "child",
            "emotion": "떨리고 설레요",
            "health": "목이 약간 아파요",
            "nutrition": "학예회 전 긴장돼서 점심을 조금에 못 먹었어요.",
            "activities": ["학예회 리허설", "학예회 공연", "친구들과 축하 파티"],
            "social": "공연 후 친구들과 서로 축하해주고 칭찬해줬어요.",
            "special": "학예회에서 솔로 파트를 맡아 노래했는데, 부모님께서 정말 자랑스러워하셨어요.",
            "keywords": ["학예회", "노래", "솔로", "축하", "파티"],
            "text": "오늘은 정말 특별한 날이었어요! 🌟\n학교 학예회가 있었는데, 제가 노래 솔로 파트를 맡았어요. 🎤\n리허설 때는 너무 떨려서 실수도 했지만, 친구들이 응원해줘서 용기를 냈어요.\n실제 공연에서는 정말 잘 불렀어요! 부모님께서 눈물을 흘리시면서 박수를 쳐주셨어요. 😊\n공연 후에는 친구들과 작은 축하 파티를 했어요. 다들 서로 칭찬하고 축하해주는 게 정말 기분 좋았어요. 🎉\n비록 목이 좀 아프지만, 오늘은 제 인생에서 가장 자랑스러운 날 중 하나예요! 💖",
        },
        {
            "date": "2024-09-04",
            "role": "parents",
            "emotion": "걱정되지만 희망적이에요",
            "health": "감기 기운이 있어요",
            "nutrition": "따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요.",
            "activities": ["재택근무", "아이 숙제 도와주기", "병원 방문"],
            "social": "화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다.",
            "special": "아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
            "keywords": ["재택근무", "숙제", "병원", "화상회의", "성장"],
            "text": "2024-09-04 걱정되지만 희망적이에요 감기 기운이 있어요 따뜻한 국물 위주로 식사했어요. 저���엔 아이와 함께 건강한 된장찌개를 끓였어요. 재택근무, 아이 숙제 도와주기, 병원 방문 화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다. 아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
        },
        {
            "date": "2024-09-04",
            "role": "child",
            "emotion": "궁금하고 걱정돼요",
            "health": "기침이 조금 나요",
            "nutrition": "입맛이 없어서 평소보다 적게 먹었어요.",
            "activities": ["온라인 수업", "책 읽기", "퍼즐 맞추기"],
            "social": "아픈 친구에게 영상통화로 안부를 물었어요.",
            "special": "혼자서 옷 단추를 다 채웠어요! 엄마가 정말 기뻐하셨어요.",
            "keywords": ["온라인수업", "책", "퍼즐", "단추", "영상통화"],
            "text": "오늘은 조금 이상한 하루였어요. 🤒\n학교에 가지 않고 집에서 온라인으로 수업을 들었어요. 선생님 얼굴을 화면으로 보는 게 신기했어요.\n기침이 나고 몸이 안 좋아서 공부하기가 조금 힘들었지만, 엄마가 계속 옆에서 도와주셨어요. 💖\n점심 먹고 나서는 새로 산 공룡 퍼즐을 맞췄어요. 어려웠지만 재미있었어요! 🦕\n저녁에는 혼자서 옷 단추를 다 채웠어요. 엄마가 정말 기뻐하시던게 기억나요. 😊\n아픈 친구한테 전화해서 괜찮은지 물어봤어요. 내일은 우리 둘 다 나아있기를 바라요. 🙏",
        },
    ]

    for sample in samples:
        document = create_document_from_data(convert_data_structure(sample))
        vector_store.add_documents(documents=[document], ids=[str(uuid4())])


# Add sample data on application startup(삭제 예정)
@router.on_event("startup")
async def startup_event():
    add_sample_data()

