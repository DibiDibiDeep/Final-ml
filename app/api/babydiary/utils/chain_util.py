import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from app.api.babydiary.models import DaycareReport
from dotenv import load_dotenv

load_dotenv()
# 환경변수 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
llm_model = os.getenv("LLM_MODEL")

# 키워드 추출 프롬프트 파일 경로
template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
keyword_prompt_path = os.path.join(template_path, "extract_keyword_prompt.txt")
write_diary_prompt_path = os.path.join(template_path, "write_diary_prompt.txt")


# 키워드 추출 체인 설정
def setup_extract_keyword_chain():
    # 프롬프트 및 파서 설정
    output_parser = JsonOutputParser(pydantic_object=DaycareReport)
    with open(keyword_prompt_path, "r", encoding="utf-8") as file:
        template = file.read()

    prompt = PromptTemplate(
        input_variables=["report"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
        template=template,
    )

    # 모델 및 체인 설정
    model = ChatOpenAI(
        model_name=llm_model, temperature=0, openai_api_key=openai_api_key
    )
    chain = prompt | model | output_parser

    return chain


# 일기 작성 체인 설정
def setup_write_diary_chain():
    # 프롬프트 및 파서 설정
    output_parser = StrOutputParser()
    with open(write_diary_prompt_path, "r", encoding="utf-8") as file:
        template = file.read()

    prompt = PromptTemplate.from_template(
        template=template,
    )
    model = ChatOpenAI(
        model_name=llm_model, temperature=0, openai_api_key=openai_api_key
    )
    chain = prompt | model | output_parser
    return chain
