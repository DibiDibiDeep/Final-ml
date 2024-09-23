from pymilvus import (
    Collection,
    connections,
    utility,
)
from .utils import insert_db

connections.connect("default", host="standalone", port="19530")

# Milvus 컬렉션 로드

if not utility.has_collection("child"):
    insert_db()
else:
    collection = Collection("child")
collection.load()
