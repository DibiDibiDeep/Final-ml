from pymilvus import (
    Collection,
    connections,
    utility,
)
from .utils import init_db
import os


# Get the MILVUS_HOST from environment variables
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
collection_name = os.getenv("COLLECTION_NAME")

connections.connect("default", host=MILVUS_HOST, port=str(MILVUS_PORT))
# Milvus 컬렉션 로드

if not utility.has_collection(collection_name):
    init_db()
else:
    collection = Collection(collection_name)
collection.load()
