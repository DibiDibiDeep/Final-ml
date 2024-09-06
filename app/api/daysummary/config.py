from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Load environment variables
load_dotenv()

# Embedding model configuration
embeddings_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Milvus vector store configuration (for local development)
URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": URI},
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
