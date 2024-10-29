import tempfile
import uuid
from urllib.parse import urlparse
import json, os
import logging
from dotenv import load_dotenv

from app.api.calendar.models import ImageInput
from app.api.calendar.BetterOCR import betterocr
from app.api.calendar.utils.s3_util import parse_s3_url, set_s3_client
from app.api.calendar.utils.chain_util import setup_chain

from fastapi import HTTPException, APIRouter

load_dotenv()

s3_client = set_s3_client()

# 체인 설정
chain = setup_chain()

router = APIRouter()


class InvalidImageTypeError(Exception):
    """Raised when the image is not a valid daycare schedule"""

    pass


@router.post("/process_image")
async def process_image(image_input: ImageInput):

    try:
        baby_id = image_input.baby_id
        user_id = image_input.user_id
        image_path = image_input.image_path
        logging.info(
            f"""
[*] Process Image Start
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
            logging.info("S3 Download Start...")
            s3_client.download_file(bucket, key, temp_file_path)
            logging.info("S3 Download End...")
            ocr_target = temp_file_path
        else:
            ocr_target = image_path

        logging.info(f"!!!Downloaded OCR target!!! : {ocr_target}")
        # Perform OCR
        logging.info("OCR Start...")
        ocr_result = betterocr.detect_text(
            ocr_target,
            ["ko", "en"],  # language codes (from EasyOCR)
            openai={
                "model": "gpt-4o-mini",
            },
        )
        logging.info("OCR End...")
        logging.info(f"\n\nOCR Result:\n{ocr_result}")

        # 임시 파일 삭제
        if is_s3:
            os.unlink(temp_file_path)
            logging.info("Temp File Delete End...")
        if not ocr_result == "Invalid image type":
            # chain을 사용하여 처리
            logging.info("LLM Generate Answer Start...")
            response = chain.invoke({"ocr_result": ocr_result})
            logging.info("LLM Generate Answer End...")
            logging.info(f"\n\nLLM Result:\n{response}")

            processed_event = response
            # json 파일에 user_id, baby_id 추가
            processed_event.update({"user_id": user_id, "baby_id": baby_id})

            return processed_event
        else:
            raise InvalidImageTypeError(
                "The provided image is not a valid daycare schedule."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
