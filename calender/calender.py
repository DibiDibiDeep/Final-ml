from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import json
import os

from util.date_util import DateProcessor

from fastapi import FastAPI, HTTPException

from langchain_teddynote import logging
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import Field

from BetterOCR import betterocr

if not os.path.exists("result"):
    os.mkdir("result")

set_llm_cache(InMemoryCache())

load_dotenv()
logging.langsmith("canlender")

# OpenAI GPT-4o-mini 모델 사용
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


class Event(BaseModel):
    date: str = Field(description="Day of the event datetime format DD(1-31)")
    activities: List[str] = Field(
        description="Concise list of activities for this date, including only essential information"
    )


class MonthlySchedule(BaseModel):
    events: List[Event] = Field(description="List of events for each date")
    etc: str = Field(
        description="Additional information that does not fit into the event list"
    )


output_parser = JsonOutputParser(pydantic_object=MonthlySchedule)

with open("prompt/calender_betterocr.txt", "r") as file:
    template = file.read()


prompt = PromptTemplate(
    input_variables=["ocr_result"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template=template,
)


# chain 생성
chain = prompt | model | output_parser

app = FastAPI()


class ImageInput(BaseModel):
    image_path: str


@app.post("/process_image")
async def process_image(image_input: ImageInput):
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

        # 결과를 json 파일로 저장(or DB 저장(추후))
        file_name = image_input.image_path.split("/")[-1].split(".")[0] + ".json"

        with open(f"./result/{file_name}", "w", encoding="utf-8") as f:
            json.dump(processed_event, f, ensure_ascii=False, indent=2)

        return processed_event

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
