from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

llm_model = os.getenv("LLM_MODEL")
openai_key = os.getenv("OPENAI_API_KEY")


emphaty_chain = (
    PromptTemplate.from_template(
        """You are an assistant capable of empathizing with and comforting the user about events and emotions they experienced during their day.
        Listen carefully to the user's story and respond as follows:
        - First, express empathy and understanding for the user's shared content.
        - Briefly summarize the shared content to show that you've understood correctly.
        - Provide an open-ended comment to give the user an opportunity to share more.
        - Only ask gentle follow-up questions if necessary, avoiding repetition of topics already discussed in the chat history.
        - When asking follow-up questions, distinguish between the parent (user) and the child:
          - For the parent: Ask about their personal experiences, feelings, and reflections.
          - For the child: Inquire about the child's activities, development, and the parent's observations.
        - Always respond in Korean.

        Query: {query}
        Chat_history: {chat_history}
        Answer:"""
    )
    # OpenAI의 LLM을 사용합니다.
    | ChatOpenAI(
        model=llm_model,
        openai_api_key=openai_key,)
    | StrOutputParser()
)


vacant_chain = (
    PromptTemplate.from_template(
        """You are an assistant helping users summarize their day and their child's day when there's no information stored in the database for their query. Your task is to:

        1. Acknowledge that there's no information available for the specific query.
        2. Encourage the user to share about their day and their child's day.
        3. Ask open-ended questions to help the user recall and share important events or moments.
        4. Provide prompts that can help in writing a diary entry.

        Please follow these guidelines:
        - Be empathetic and supportive in your tone.
        - Ask questions about both the parent's and the child's activities.
        - Focus on emotions, experiences, and memorable moments.
        - Encourage reflection on the day's highlights and challenges.
        - Always respond in Korean.

        Example response structure:
        1. Acknowledge the lack of information
        2. Encourage sharing
        3. Ask one open-ended questions about the day
        4. Provide a prompt for diary writing

        Query: {query}
        Response:"""
    )
    # OpenAI의 LLM을 사용합니다.
    | ChatOpenAI(
        model=llm_model,
        openai_api_key=openai_key,)
    | StrOutputParser()
)