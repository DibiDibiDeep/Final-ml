from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import vector_store


@tool
def classify_intent(query: str) -> str:
    """
    Classifies the intent of the user's query into more specific categories.
    Use this tool to determine the user's intention and guide the conversation flow.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Classify the user's query into one of these categories:
                'QUESTION_CHILD': for queries about the child's day or activities
                'QUESTION_PARENT': for queries about the parent's day or activities
                'DIARY_REQUEST': for requests to write a diary entry
                'OTHER': for expressions of emotion or mood
                
                Provide only the category name as the response.""",
            ),
            ("user", "{query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return response.content.strip().upper()


@tool
def parent_retriever_assistant(query: str) -> str:
    """
    Retrieves information about the parent's day and activities and generates a response.
    Use this tool when you need to find specific information about the parent's past events.
    Retriever filter: role = parents
    """
    parents_result = vector_store.similarity_search(
        query,
        k=1,
        expr="role == 'parents'",
    )
    return parents_result


@tool
def child_retriever_assistant(query: str) -> str:
    """
    Retrieves information about the child's day and activities and generates a response.
    Use this tool when you need to find specific information about the child's past events.
    Retriever filter: role = child
    """
    child_result = vector_store.similarity_search(
        query,
        k=1,
        expr="role == 'child'",
    )
    return child_result
