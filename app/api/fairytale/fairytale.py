from dotenv import load_dotenv
import os
import time
import logging


from app.api.fairytale.utils.fairytale_utils import (
    select_keys_from_diary_data,
    create_pairy_chain,
    save_result_to_json
)
from app.api.fairytale.utils.image_generation_utils import (
    generate_image,
    create_image_prompt
)
from app.api.fairytale.utils.image_cut_utils import (
    detect_and_crop_panels,
    numpy_to_base64,
)
from app.api.fairytale.models import FairytaleInput
import cv2
from openai import AsyncOpenAI

from fastapi import APIRouter, HTTPException

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
router = APIRouter()


@router.post("/generate_fairytale")
async def generate_fairytale(input_data: FairytaleInput):
    # try:
    start_time = time.time()
    # 모델 인스턴스 데이터 -> 딕셔너리 데이터로 변환
    data = input_data.model_dump()
    user_id = data["user_id"]
    baby_id = data["baby_id"]

    # Request 데이터 전처리(필요한 데이터만 추출)
    data = select_keys_from_diary_data(data)
    # 프롬프트 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "prompts", "fairytale_prompt_ver4.txt")

    with open(
        prompt_path,
        "r",
        encoding="utf-8",
    ) as file:
        prompts = file.read()

    # 동화 생성 체인 생성
    chain = create_pairy_chain(prompts, data)

    logging.info("Fairytale generation started")
    # 동화 생성
    result = chain.invoke(
        {
            "name": data["name"],
            "age": data["age"],
            "gender": data["gender"],
            "activities": data["activities"],
            "special": data["special"],
        }
    )
    logging.info("Fairytale generation completed")

    dall_e_prompt = create_image_prompt(result)
    print(dall_e_prompt)
    # # 이미지 생성 클라이언트 생성
    client = AsyncOpenAI(api_key=apikey)
    url = await generate_image(client, dall_e_prompt)
    logging.info(f"Generate Image URL: {url}")

    # 이미지 다운로드 및 패널로 분할
    panels = detect_and_crop_panels(url)
    # 패널 이미지 base64 인코딩
    base64_panels = [numpy_to_base64(panel) for panel in panels]

    # 결과 표지 이미지 추가
    result['cover_illustration'] = base64_panels.pop(0)
    # 페이지별 이미지 추가
    for page, panel in zip(result["pages"], base64_panels):
        page["illustration"] = panel
        
    # 결과에 사용자 ID, 아기 ID 추가
    result.update({"user_id": user_id, "baby_id": baby_id})
    # 총 소요 시간 계산
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time: {total_time:.2f} seconds")

    return result
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
