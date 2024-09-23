import os
import logging
from datetime import datetime

from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    RRFRanker,
)
from openai import OpenAI
embedding_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embedding(client, text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


@tool
def classify_intent(query: str) -> str:
    """
    This function classifies the intent of a given query.
    
    Parameters:
    query (str): The query input by the user
    
    Returns:
    str: 'QUESTION' or 'DIARY_REQUEST' or 'OTHER'
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", 
                     openai_api_key=os.getenv("OPENAI_API_KEY"),
                     temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Classify the user's query into one of these categories:
                'QUESTION': for inquiries about specific events, activities, or details of the parent's or child's day. This includes asking about what happened, who was involved, or requesting factual information about their experiences.
                'DIARY_REQUEST': for explicit requests to write or compose a diary entry, journal entry, or reflective piece about the day's events. This category is used when the user asks for a written account or summary of the day, rather than asking for specific information.
                'OTHER': for expressions of emotion, mood, or any input that doesn't fit into the above categories.
                Provide only the category name as the response.
                """,
            ),
            ("user", "{query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return response.content.strip().upper()


# 하이브리드 검색 함수
@tool
def retreiver_about_qeustion(query: str, expr: str):
    """
    Retrieves information about the parent's or child's day and activities and generates a response.
    Use this tool when you need to find specific information about the parent's or child's past events.
    """
    today_date = datetime.now().strftime('%Y-%m-%d')
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.0,
        )
    """
        - emotion (VARCHAR)
        - health (VARCHAR)
        - nutrition (VARCHAR)
        - activities (VARCHAR)
        - social (VARCHAR)
        - special (VARCHAR)
        - keywords (VARCHAR)
    """
    template = """
        Given the following search query: "{query}"
    Generate a Milvus expression to filter the search results. The expression should be based on the fields available in the collection:
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
    - Query: "What did I eat for dinner today?"
    - Expression: date == '{today_date}' and role == 'parents'

    Example 2:
    - Query: "Did I go to the park with my friends today?"
    - Expression: role == 'parents'
    """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    # 쿼리 표현식 생성
    expr = chain.invoke({"query": query, "today_date": today_date})
    logging.info(f"Generated expression: {expr}")
    # 쿼리 임베딩
    query_embeddings = get_embedding(embedding_client, query)

    res = collection.search(
        [query_embeddings],
        expr = expr,
        anns_field='embedding', 
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},  # COSINE 메트릭 타입 사용
        rerank=RRFRanker(), 
        limit=1, 
        output_fields=['text']
    )
    logging.info(f"Search result: {res}")
    try:
        result = res[0][0].get('text')
        return result
    except:
        result = "I don't know"
        logging.info(f"Search result: {result}")
        return result
