from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from langchain.agents import tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_teddynote import logging

from dotenv import load_dotenv
from util.preprocess import convert_data_structure, create_document_from_data
from uuid import uuid4

logging.langsmith("DaySummarization Agent")
load_dotenv()

# 임베딩 모델 설정
embeddings_model = HuggingFaceEmbeddings(
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


# Tool 정의
@tool
def answer_assistant(query: str) -> str:
    """
    Retrieves relevant documents based on the given query and generates an answer using a language model.
    """
    milvus_retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1}
    )
    results = milvus_retriever.get_relevant_documents(query)

    if results:
        context = results[0].page_content
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a very powerful assistant. Answer the following question based on the given information.\nInformation: {context}",
                ),
                ("user", "{query}"),
            ]
        )

        chain = prompt | llm
        response = chain.invoke({"context": context, "query": query})
        return response.content
    else:
        return "죄송합니다. 관련 정보를 찾을 수 없습니다."


# LLM 프롬프트 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, Answer the following question based on the given information.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# LLM 모델 설정
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Tool 리스트 정의
tools = [answer_assistant]

# LLM 모델과 tool binding
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
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# AgentExecutor 설정
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# FastAPI 설정
app = FastAPI()


# 질문 데이터 형식 정의
class Question(BaseModel):
    text: str


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


# 질문 API
@app.post("/ask")
async def ask_question(question: Question):
    try:
        result = agent_executor.invoke({"input": question.text})
        return {"answer": result["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 일기 추가 API(일기 데이터를 Milvus에 추가, DB 연결하면 삭제 예정)
@app.post("/add_diary")
async def add_diary(diary: DiaryEntry):
    try:
        document = create_document_from_data(convert_data_structure(diary.dict()))
        vector_store.add_documents(documents=[document], ids=[str(uuid4())])
        return {"message": "Diary entry added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 샘플 데이터 추가 함수
def add_sample_data():
    samples = [
        {
            "name": "지수",
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
            "diary": "오늘은 정말 즐거운 하루였어! 😄  \n아침에 친구들이랑 아이스크림 가게 역할놀이를 했어. 🍦  \n나는 아이스크림을 팔고, 친구들은 손님이 되었지!  \n아이스크림 먹는 연기를 정말 잘했어! 😋  \n그 다음에는 놀이터에 가서 신나게 놀았어. 🛝  \n미끄럼틀도 타고, 그네도 타고, 정말 재밌었어!  \n친구들과 함께 웃음소리가 끊이지 않았어. 😂  \n마지막으로 붓으로 그림을 그렸는데, 너무 즐거웠어! 🎨  \n그림을 그리면서도 계속 웃고 있었어.  \n오늘 하루가 너무 행복했어! 💖",
        },
        {
            "name": "지수",
            "emotion": "처음에는 조금 슬펐지만, 친구들과 함께 하면서 금세 행복해졌어요.",
            "health": "전반적으로 건강해 보였고, 활동적인 하루를 보냈어요.",
            "nutrition": "점심시간에 처음에는 먹고 싶지 않다고 했지만, 선생님이 도와주신 덕분에 거의 다 먹었어요. 브로콜리와 메추리알은 안 먹었어요.",
            "activities": [
                "차량에서 내리기",
                "교실에서 친구들에게 간식 나누기",
                "당나귀 타기",
                "점심 먹기",
            ],
            "social": "친구들과 간식을 나누고, 당나귀를 타면서 즐겁게 놀았어요.",
            "special": "당나귀를 처음 보고 무서워했지만, 친구가 타는 모습을 보고 용기를 내서 멋지게 탔어요.",
            "keywords": [
                "슬픔",
                "행복",
                "당나귀",
                "간식",
                "점심",
                "브로콜리",
                "메추리알",
            ],
            "diary": "오늘은 정말 신나는 하루였어요! 😊  \n처음에는 조금 슬펐지만, 친구들과 함께 하면서 금세 행복해졌어요. 😄  \n전반적으로 건강해 보였고, 활동적인 하루를 보냈어요. 💪  \n점심시간에 처음에는 먹고 싶지 않다고 했지만, 선생님이 도와주신 덕분에 거의 다 먹었어요. 🍽️  \n브로콜리와 메추리알은 안 먹었어요. 😅  \n차량에서 내리자마자 친구들에게 간식을 나눠줬어요. 🍪  \n당나귀를 처음 보고 무서워했지만, 친구가 타는 모습을 보고 용기를 내서 멋지게 탔어요! 🐴  \n당나귀 타는 건 정말 재미있었어요! 🎉  \n친구들과 함께 놀면서 정말 즐거운 시간을 보냈어요. 💖  \n오늘 하루는 정말 특별했어요! 🌟",
        },
        {
            "name": "지수",
            "emotion": "즐거움과 자신감",
            "health": "좋음",
            "nutrition": "점심으로 맛있는 밥과 반찬을 먹었어요.",
            "activities": [
                "동화 '스스로 할 수 있어요' 읽기",
                "내가 할 수 있는 일 이야기하기",
                "활동지를 통해 그림 보고 선택하기",
                "선택한 일에 대해 언어로 표현하기",
            ],
            "social": "친구들과 함께 이야기하며 즐거운 시간을 보냈어요.",
            "special": "내가 할 수 있는 일들을 스스로 선택하고 표현하는 것이 정말 뿌듯했어요.",
            "keywords": ["동화", "자신감", "활동지", "표현", "친구들"],
            "diary": "오늘은 정말 즐거운 하루였어요! 😊  \n아침에 일어나서 기분이 좋았어요.  \n점심으로 맛있는 밥과 반찬을 먹었어요. 🍚🥗  \n그 후에 동화 '스스로 할 수 있어요'를 읽었어요. 📖  \n이 동화는 내가 할 수 있는 일에 대해 이야기하는 거였어요.  \n친구들과 함께 이야기하며 즐거운 시간을 보냈어요! 👫  \n활동지를 통해 그림을 보고 선택했어요. 🎨  \n내가 선택한 일에 대해 언어로 표현하는 것도 정말 재미있었어요.  \n내가 할 수 있는 일들을 스스로 선택하고 표현하는 것이 정말 뿌듯했어요! 🌟  \n오늘 하루가 너무 행복했어요! 💖",
        },
        {
            "name": "지수",
            "emotion": "행복하고 자랑스러워요",
            "health": "아주 건강해요",
            "nutrition": "오늘의 식사는 맛있었어요. 무엇을 먹었는지는 보고서에 없지만, 잘 먹었어요!",
            "activities": ["체조", "공놀이"],
            "social": "친구들과 함께 놀면서 양보도 많이 했어요. 친구가 힘들어할 때 도와주었답니다.",
            "special": "어제 친구와 인형 때문에 싸웠지만, 오늘은 양보하기로 했어요. 친구가 힘들어할 때 고민하다가도 양보해주었어요. 칭찬 많이 해주세요!",
            "keywords": ["체조", "공놀이", "양보", "친구", "인형"],
            "diary": "오늘은 정말 행복하고 자랑스러워요! 😊  \n아주 건강하게 지내고 있어요. 💪  \n오늘의 식사는 맛있었어요! 🍽️  \n무엇을 먹었는지는 모르지만, 잘 먹었어요! 😋  \n\n체조도 하고 공놀이도 했어요! 🤸‍♂️⚽  \n친구들과 함께 놀면서 양보도 많이 했어요. 🤗  \n친구가 힘들어할 때 도와주었답니다. 🥰  \n\n어제 친구와 인형 때문에 싸웠지만, 오늘은 양보하기로 했어요. 🤝  \n친구가 힘들어할 때 고민하다가도 양보해주었어요.  \n칭찬 많이 해주세요! 🌟",
        },
        {
            "name": "지수",
            "emotion": "즐거움과 호기심",
            "health": "활기차고 건강함",
            "nutrition": "오늘 특별한 호박 요리가 많았어요.",
            "activities": [
                "색깔판을 이용한 교실 탐험",
                "단호박 퍼즐 맞추기",
                "비닐로 풍선 만들기",
                "비닐터널 지나기",
                "비 표현하기",
            ],
            "social": "선생님과의 상호작용이 즐거웠고, 친구들과 함께 놀이를 하며 소통했어요.",
            "special": "자유롭게 퍼즐을 맞추며 창의력을 발휘했어요.",
            "keywords": ["호박요리", "퍼즐", "소꿉놀이", "상호작용", "비닐터널"],
            "diary": "오늘은 정말 즐거운 하루였어요! 😊  \n아침에 일어나서 기분이 너무 좋았어요.  \n오늘은 특별한 호박 요리가 많았어요! 🎃  \n색깔판을 이용해서 교실을 탐험했어요.  \n색깔이 정말 예쁘고 신기했어요! 🌈  \n단호박 퍼즐도 맞췄는데, 너무 재밌었어요! 🧩  \n자유롭게 퍼즐을 맞추면서 창의력을 발휘했어요.  \n비닐로 풍선도 만들었고, 정말 신났어요! 🎈  \n그리고 비닐터널도 지나갔어요.  \n터널 속에서 친구들과 함께 놀았어요!  \n선생님과의 상호작용도 즐거웠고,  \n친구들과 소통하며 놀 수 있어서 행복했어요. 🤗  \n오늘 하루는 정말 신나는 하루였어요! 🌟",
        },
    ]

    for sample in samples:
        document = create_document_from_data(convert_data_structure(sample))
        vector_store.add_documents(documents=[document], ids=[str(uuid4())])


# 애플리케이션 시작 시 샘플 데이터 추가
@app.on_event("startup")
async def startup_event():
    add_sample_data()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
