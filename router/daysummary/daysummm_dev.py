from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

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
from util.preprocess import convert_data_structure, create_document_from_data
from uuid import uuid4

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

app = FastAPI()

milvus_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
rag_assistant = create_retriever_tool(
    milvus_retriever,
    "search_information",
    """
    Retrieves a concise answer based on the given query using RAG.
    Use this tool when you need to find specific information about past events.
    """,
)


## 추가해야하는 기능: chat_history를 통해 사용자 답변을 포함하여 일기 작성.
@tool
def diary_writer(context: str) -> str:
    """
    Generates a brief diary entry based on the given context.
    Use this tool when you need to summarize a full day's activities.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that writes brief diary entries. Based on the given context, write a short diary entry (3-5 sentences) summarizing the key events or information.",
            ),
            ("user", "{context}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"context": context})
    return response.content


# 사용자 질문 의도 파악(QUESTION, SUMMARY, OTHER / 추가 기능해야하는 기능:부모 or 아이에 관한 질의인지)
@tool
def classify_intent(query: str) -> str:
    """
    Classifies the intent of the user's query.
    Use this tool to determine whether the user is asking a specific question or requesting a summary.
    """
    llm = ChatOpenAI(model_name="gpt-4-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify the user's query into one of these categories: 'QUESTION' (for queries about specific information), 'SUMMARY' (for requests to summarize the day), or 'OTHER' (for anything else).",
            ),
            ("user", "{query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return response.content.strip().upper()


# 대화 당일 하루 정보를 통해 일기 작성을 도와주는 질문 생성
@tool
def question_generator():
    """
    Extracts today's user events for the question generation tool.
    """
    # 임의(디비에서 현재 날짜 사용자 ID기준으로 데이터 추출 과정 필요)
    today_user_events = {
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
                f"""You are an empathetic AI assistant that engages parents in in-depth conversations about their child's day.
                Based on the following recent diary entries, parents can reflect more deeply on themselves and their children's days. 
                Create one question based on the user's daily schedule to help you understand each other's experiences.
                Questions should be specific, focus on the individual's experiences and feelings, and be approached from a positive perspective.
                Always provide translation into Korean.
                """,
            ),
            ("user", "User's Today Events:{query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": today_user_events["text"]})
    return response.content


# Agent 프롬프트 수정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant designed to help with questions about a person's day and summarize daily activities. Follow these steps:\n"
            "1. Use the classify_intent tool to determine the user's intent.\n"
            "2. If it's a QUESTION, use the answer_assistant tool to provide a concise answer.\n"
            "3. If it's a SUMMARY request, use the diary_writer tool to generate a brief summary.\n"
            "4. For OTHER intents, generate a question that helps organize emotions about specific events extracted from the user's day.\n"
            "Always provide only the necessary information without extra steps or explanations.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# LLM 및 tools 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
tools = [classify_intent, rag_assistant, diary_writer, question_generator]

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Agent 설정
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    | OpenAIFunctionsAgentOutputParser()
)

# AgentExecutor 설정
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 일기 데이터 형식 정의
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


# AgentExecutor 설정
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


class Query(BaseModel):
    text: str


# API 엔드포인트
@app.post("/process_query")
async def process_user_query(query: Query):
    try:
        result = agent_executor.invoke({"input": query.text})
        return {"response": result["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_diary")
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
            "activities": ["아이 등하원", "업무", "저녁 산책"],
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
            "nutrition": "아침은 오트밀, 점심은 회사 구내식당, 저녁은 아이와 함께 삼계탕을 먹었어요.",
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
            "nutrition": "학교 급식을 맛있게 먹었어요. 특히 디저트가 맛있었대요.",
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
            "nutrition": "학예회 전 긴장돼서 점심을 조금밖에 못 먹었어요.",
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
            "text": "2024-09-04 걱정되지만 희망적이에요 감기 기운이 있어요 따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요. 재택근무, 아이 숙제 도와주기, 병원 방문 화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다. 아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
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
@app.on_event("startup")
async def startup_event():
    add_sample_data()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
