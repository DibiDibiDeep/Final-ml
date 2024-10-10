from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    utility,
)
import logging

# Milvus 컬렉션 생성 함수
def prepare_vecdb(collection_name, embedding_dimension, milvus_host, milvus_port):
    # Milvus 연결 및 컬렉션 생성
    connections.connect("default", host=milvus_host, port=str(milvus_port))

    # 컬렉션 생성
    if utility.has_collection(collection_name):
        # Collection 있을 경우 불러오기
        logging.info(f"Collection '{collection_name}' already exists. Skipping data insertion.")
        collection = Collection(collection_name)
        
    else:
        # Collection 없을 경우 생성.
        # Schema 생성
        logging.info(f"Collection '{collection_name}' does not exist. Creating new collection.")
        fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="baby_id", dtype=DataType.INT64),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimension),
            ]

        # 컬렉션 schema 생성
        schema = CollectionSchema(
            fields, "Collection for storing text and embeddings about child and parents")
        
        collection = Collection(collection_name, schema)

        # 인덱스 생성
        collection.create_index(
            "embedding", {"index_type": "FLAT", "metric_type": "COSINE"}
            # index_type 데이터 수가 늘어남에 따라 다양한 종류 고려 필요.
        )
    return collection