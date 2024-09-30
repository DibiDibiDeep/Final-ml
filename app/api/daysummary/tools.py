import os
import logging
from datetime import datetime
import requests

from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import collection
from pymilvus import (
    RRFRanker,
)
from openai import OpenAI

openai_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

embedding_client = OpenAI(api_key=openai_key)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_embedding(client, text, model=embedding_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# 하이브리드 쿼리 검색 함수
@tool
def retreiver_about_qeustion(user_id: int, baby_id: int, query: str):
    """
    Use this tool when you need to find specific information about the parent's or child's events.
    Retrieves information about the parent's or child's day and activities and generates a response.

    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    logging.info(f"Input parameters - user_id: {user_id}, baby_id: {baby_id}, query: {query}")
    llm = ChatOpenAI(
        openai_api_key=openai_key,
        model= llm_model,
        temperature=0.0,
    )
    
    template = """
        Given the following search query: "{query}"
    Generate a Milvus expression to filter the search results. The expression should be based on the fields available in the collection:
    - user_id (INT64){user_id}, baby_id (INT64){baby_id}
    - date (VARCHAR, format: "YYYY-MM-DD")
      - If the user mentions "today", use today's date ({today_date}) to generate the response.
      - In other cases, do not use today's date.
      - If the date cannot be determined from the user's query, only generate the expression for the role.
    - role (VARCHAR)
      - role is 'parents' or 'child'
      - If the query is about the user's activities, use role == 'parents'
      - If the query is about the user's child's activities, use role == 'child'
    If two expressions need to be used, connect the expressions with 'and'.
    Return only the expression, without any explanation, additional text, or backticks.
    
    Example 1:
    - Query: user_id: {user_id}, baby_id: {baby_id}, "What did I eat for dinner today?"
    - Expression: user_id == {user_id} and baby_id == {baby_id} and date == '{today_date}' and role == 'parents'

    Example 2:
    - Query: user_id: {user_id}, baby_id: {baby_id}, "Did I go to the park with my friends today?"
    - Expression: user_id == {user_id} and baby_id == {baby_id} and date == '{today_date}' and role == 'parents'
    """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    # 쿼리 표현식 생성
    expr = chain.invoke({"query": query, "today_date": today_date, "user_id": user_id, "baby_id": baby_id})
    logging.info(f"Generated expression: {expr}")
    # 쿼리 임베딩
    query_embeddings = get_embedding(embedding_client, query)

    res = collection.search(
        [query_embeddings],
        expr=expr,
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        },  # COSINE 메트릭 타입 사용
        rerank=RRFRanker(),
        limit=1,
        output_fields=["date", "text"],
    )
    logging.info(f"Search result: {res}")
    print(res[0])
    if len(res[0]) == 0:
        return "No results found"
    else:
        return res
    

@tool
def save_diary(user_id: int, baby_id: int, content: str) -> bool:
    """Use this when you want to POST to a Backend API.
    Be careful to always use double quotes for strings in the json string
    The output will be the text response of the POST request.
    If the response is successful, the function will return true.
    After a successful response, stop using this tool and inform the user about the successful save.
    """
    backend_url = os.getenv("BACKEND_API_URL")
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "userId": user_id,
        "babyId": baby_id,
        "content": content,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    logging.info(f"Data to be saved: {data} ,Data type: {type(data)}")
    try:
        response = requests.post(backend_url + "/api/today-sum", json=data, headers=headers)
        # logging.info(f"Response from backend: {data}")
        return True
    except requests.exceptions.RequestException as e:
        # logging.error(f"Error saving diary: {str(e)}")
        return False