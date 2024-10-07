from dotenv import load_dotenv
import base64
import aiohttp
import asyncio
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
)
from app.api.fairytale.utils.image_cut_utils import detect_and_crop_panels
from app.api.fairytale.models import FairytaleInput
import cv2
from openai import AsyncOpenAI

from fastapi import APIRouter, HTTPException

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
router = APIRouter()

def numpy_to_base64(image):
    # OpenCV uses BGR, convert to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Encode image to jpg
    _, buffer = cv2.imencode('.jpeg', image)
    # Convert to base64 string
    return base64.b64encode(buffer).decode('utf-8')

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
    prompt_path = os.path.join(current_dir, "prompts", "fairytale_prompt_ver3.txt")

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
            # "art_style": data["art_style"],
            "art_style": "Cartoon animated",
        }
    )
    print(result)
    logging.info("Fairytale generation completed")

    # 캐릭터 정보를 문자열로 변환
    characters_info = " ".join(
        [
            f"{char['name']}: {char['description']}"
            for char in result.get("characters", [])
        ]
    )
    # 페이지 텍스트와 일러스트레이션 프롬프트 생성
    all_page_text = "\n".join(page["text"] for page in result["pages"])
    all_page_image_prompt = "\n".join(page["illustration_prompt"] for page in result["pages"])

    # 최종 이미지 프롬프트 생성
    all_page_image_prompt = "\n".join(
        [
        f"Characters information: {characters_info}",
        result["book_cover_description"],
        all_page_image_prompt,
        """
        Generate image a grid that consists of 4 panels each panel image must be 512 by 512 pixels. 
        Panel image order is left to right, top to bottom. 
        Do not include any text in the images.
        """
        ]
    )
    # # 이미지 생성 클라이언트 생성
    client = AsyncOpenAI(api_key=apikey)
    url = await generate_image(client, all_page_image_prompt)
    logging.info(f"Generate Image URL: {url}")

    panels = detect_and_crop_panels(url)
    base64_panels = [numpy_to_base64(panel) for panel in panels]
    result['cover_illustration'] = base64_panels.pop(0)

    for page, panel in zip(result["pages"], base64_panels):
        page["illustration"] = panel
    print(result)
    # 총 소요 시간 계산
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time: {total_time:.2f} seconds")
    # print(result)
    

    # # JSON 파일로 결과 저장
    save_result_to_json(result, "fairytale_result_base64.json")
    logging.info("JSON file saved")

    # return result
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
