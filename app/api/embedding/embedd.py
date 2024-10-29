import os
from openai import OpenAI
import logging
from fastapi import APIRouter, HTTPException
from .models import DayInfoBatch
from .utils.vecdb_util import prepare_vecdb

# 환경변수 설정
openai_key = os.getenv("OPENAI_API_KEY")
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")
embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_dimension = os.getenv("EMBEDDING_DIMENSION")
collection_name = os.getenv("COLLECTION_NAME")


# Embedding Client 설정
client = OpenAI(api_key=openai_key)

# Milvus 연결 및 컬렉션 생성
collection = prepare_vecdb(collection_name, embedding_dimension, milvus_host, milvus_port)

# 임베딩 생성 함수
def get_embedding(client, text, model= embedding_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

router = APIRouter()

# DayInfoBatch 데이터를 받아서 임베딩 생성 후 Milvus에 저장.
@router.post("/embedding")
async def insert_day_info(batch: DayInfoBatch):
    try:
        entities = []
        for item in batch.items:
            entity = {
                "user_id": item.user_id,
                "baby_id": item.baby_id,
                "date": item.date,
                "role": item.role,
                "text": item.text,
                "embedding": get_embedding(client, item.text),
            }
            entities.append(entity)
        logging.info(f"""
Successfully created entities\n
user_id: {entities[0]['user_id']}\n
baby_id: {entities[0]['baby_id']}\n
date: {entities[0]['date']}\n
role: {entities[0]['role']}\n
text: {entities[0]['text']}\n
"""
)
        collection.insert(entities)
        collection.flush()
        logging.info(f"Successfully inserted {len(entities)} entities into the collection\n Total Collection Size: {collection.num_entities}")
        return {"message": f"Successfully inserted {len(entities)} entities into the collection\n Total Collection Size: {collection.num_entities}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
