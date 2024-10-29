import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.api.calendar.models import MonthlySchedule
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
llm_model = os.getenv("LLM_MODEL")

template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
calendar_prompt_path = os.path.join(template_path, "calendar_betterocr_ver3.txt")


def setup_chain():
    # 파서 정의
    output_parser = JsonOutputParser(pydantic_object=MonthlySchedule)

    # 프롬프트 템플릿 불러오기
    with open(calendar_prompt_path, "r", encoding="utf-8") as file:
        template = file.read()

    # 프롬프트 정의
    prompt = PromptTemplate(
        input_variables=["ocr_result"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
        template=template,
    )
    # LLM 모델 정의
    model = ChatOpenAI(
        model_name=llm_model, temperature=0, openai_api_key=openai_api_key
    )
    return prompt | model | output_parser
