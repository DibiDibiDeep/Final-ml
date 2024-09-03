from fastapi import HTTPException, APIRouter

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from models.calendar import MonthlySchedule, ImageInput
from .BetterOCR import betterocr
from .utils.date_util import DateProcessor

set_llm_cache(InMemoryCache())

# OpenAI GPT-4o-mini 모델 사용
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
output_parser = JsonOutputParser(pydantic_object=MonthlySchedule)

with open("api/calendar/prompts/calendar_betterocr.txt", "r", encoding="utf-8") as file:
    template = file.read()

prompt = PromptTemplate(
    input_variables=["ocr_result"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template=template,
)

# chain 생성
chain = prompt | model | output_parser

router = APIRouter()


@router.post("/process_image")
async def process_image(image_input: ImageInput):
    from langchain_teddynote import logging

    logging.langsmith("canlender")

    try:

        # Perform OCR
        print("OCR Start...")
        ocr_result = betterocr.detect_text(
            image_input.image_path,
            ["ko", "en"],  # language codes (from EasyOCR)
            openai={
                "model": "gpt-4o-mini",
            },
        )
        print("OCR End...")

        # chain을 사용하여 처리
        print("LLM Start...")
        response = chain.invoke({"ocr_result": ocr_result})
        print("LLM End...")

        # DateProcessor를 사용하여 최종 처리
        print("LLM Result Postprocessing Start...")
        # json key값 date 형식 정규화(11일 -> 11, 11일 화 -> 11)
        event_processor = DateProcessor(response)
        processed_event = event_processor.process()
        print("LLM Result Postprocessing End...")

        return processed_event

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))