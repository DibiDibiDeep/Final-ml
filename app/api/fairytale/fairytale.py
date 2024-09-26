from app.api.fairytale.utils.fairytale_utils import (
    select_keys_from_diary_data,
    create_pairy_chain,
    save_result_to_json,
)
from app.api.fairytale.utils.fairytale_image_utils import generate_all_images
from app.api.fairytale.models import FairytaleInput

from dotenv import load_dotenv
import os
import time
import logging
from openai import AsyncOpenAI

from fastapi import APIRouter, HTTPException

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
router = APIRouter()


@router.post("/generate_fairytale")
async def generate_fairytale(input_data: FairytaleInput):
    try:
        start_time = time.time()
        # 모델 인스턴스 데이터 -> 딕셔너리 데이터로 변환
        data = input_data.model_dump()
        user_id = data["user_id"]
        baby_id = data["baby_id"]


        # Request 데이터 전처리(필요한 데이터만 추출)
        data = select_keys_from_diary_data(data)
        # 프롬프트 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "prompts", "fairytale_prompt_ver2.txt")

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
        logging.info("Fairytale generation completed")

        # 캐릭터 정보를 문자열로 변환
        characters_info = " ".join(
            [
                f"{char['name']}: {char['description']}"
                for char in result.get("characters", [])
            ]
        )
        # 이미지 생성 클라이언트 생성
        client = AsyncOpenAI(api_key=apikey)
        
        # 이미지 생성
        cover_image, page_images = await generate_all_images(
            client, 
            characters_info, 
            result["book_cover_description"], 
            result["pages"]
        )

        # 결과 처리
        result["title_img_path"] = cover_image
        result["pages"] = [{"image_url": img[1]["image_url"], **img[1]} for img in sorted(page_images, key=lambda x: x[0])]
        

        result.update({"user_id": user_id, "baby_id": baby_id})
        logging.info(f"Result: {result}")
        # 총 소요 시간 계산
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Total time: {total_time:.2f} seconds")

        # JSON 파일로 결과 저장
        save_result_to_json(result, "fairytale_result_fix_timeout_ver3_1.json")
        logging.info("JSON file saved")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

