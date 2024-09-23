from pymilvus import (
    connections,
    Collection
)
from .utils import insert_db

connections.connect("default", host="standalone", port="19530")


# Milvus 컬렉션 로드
try:
    collection = Collection("child")
except Exception as e:
    insert_db()
    collection = Collection("child")
collection.load()