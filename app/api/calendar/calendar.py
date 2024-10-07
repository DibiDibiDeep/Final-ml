from dotenv import load_dotenv

load_dotenv()

from fastapi import HTTPException, APIRouter
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


from app.api.calendar.models import MonthlySchedule, ImageInput
from app.api.calendar.BetterOCR import betterocr

from app.api.calendar.utils.s3_util import parse_s3_url, set_s3_client
import tempfile
import uuid
from urllib.parse import urlparse, unquote
import json, os

set_llm_cache(InMemoryCache())
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# OpenAI GPT-4o-mini 모델 사용
model = ChatOpenAI(
    model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key
)
output_parser = JsonOutputParser(pydantic_object=MonthlySchedule)

with open(
    "app/api/calendar/prompts/calendar_betterocr_ver3.txt", "r", encoding="utf-8"
) as file:
    template = file.read()

prompt = PromptTemplate(
    input_variables=["ocr_result"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template=template,
)

# chain 생성
chain = prompt | model | output_parser

s3_client = set_s3_client()

router = APIRouter()


class InvalidImageTypeError(Exception):
    """Raised when the image is not a valid daycare schedule"""

    pass


@router.post("/process_image")
async def process_image(image_input: ImageInput):
    from langchain_teddynote import logging

    # logging.langsmith("canlender")

    try:
        baby_id = image_input.baby_id
        user_id = image_input.user_id
        image_path = image_input.image_path
        print(
            f"""
              baby_id: {baby_id}
              user_id: {user_id}
              image_path: {image_path}
              """
        )

        # S3 URL 감지
        parsed_url = urlparse(image_path)
        is_s3 = parsed_url.netloc.endswith("amazonaws.com")
        if is_s3:
            # 안전한 임시 파일 이름 생성
            temp_file_name = f"{uuid.uuid4().hex}.jpg"
            temp_file_path = os.path.join(tempfile.gettempdir(), temp_file_name)

            bucket = os.getenv("AWS_S3_BUCKET")
            key = parse_s3_url(image_path)["full_file_name"]

            # s3에서 이미지 다운로드
            print("S3 Download Start...")
            s3_client.download_file(bucket, key, temp_file_path)
            print("S3 Download End...")
            ocr_target = temp_file_path
        else:
            ocr_target = image_path

        print(f"!!!Downloaded OCR target!!! : ", ocr_target)
        # Perform OCR
        print("OCR Start...")
        ocr_result = betterocr.detect_text(
            ocr_target,
            ["ko", "en"],  # language codes (from EasyOCR)
            openai={
                "model": "gpt-4o-mini",
            },
        )
        print("\n\nOCR Result:")
        print(ocr_result)
        print("OCR End...")

        # 임시 파일 삭제
        if is_s3:
            os.unlink(temp_file_path)
            print("Temp File Delete End...")
        if not ocr_result == "Invalid image type":
            # chain을 사용하여 처리
            print("LLM Start...")
            response = chain.invoke({"ocr_result": ocr_result})
            print("LLM End...")

            # DateProcessor를 사용하여 최종 처리
            print("LLM Result Postprocessing Start...")
            # json key값 date 형식 정규화(11일 -> 11, 11일 화 -> 11)
            # event_processor = DateProcessor(response)
            # processed_event = event_processor.process()
            print("LLM Result Postprocessing End...")

            processed_event = response
            # json 파일에 user_id, baby_id 추가
            processed_event.update({"user_id": user_id, "baby_id": baby_id})

            # 결과를 json 파일로 저장(or DB 저장(추후))
            file_name = image_path.split("/")[-1].split(".")[0] + ".json"
            if not os.path.exists("results/"):
                os.makedirs("results/")
            with open(f"results/{file_name}", "w", encoding="utf-8") as f:
                json.dump(processed_event, f, ensure_ascii=False, indent=2)

            return processed_event
        else:
            raise InvalidImageTypeError(
                "The provided image is not a valid daycare schedule."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
