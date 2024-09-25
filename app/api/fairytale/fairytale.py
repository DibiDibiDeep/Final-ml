from app.api.fairytale.utils.fairytale_utils import (
    select_keys_from_diary_data,
    create_pairy_chain,
    save_result_to_json,
)
from app.api.fairytale.utils.fairytale_image_utils import generate_cover_image, generate_page_image
from app.api.fairytale.models import FairytaleInput

from dotenv import load_dotenv
from tqdm import tqdm
import os
import time
import asyncio

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

        print("동화 생성 시작")
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
        print("동화 생성 완료")
        # 캐릭터 정보를 문자열로 변환
        characters_info = " ".join(
            [
                f"{char['name']}: {char['description']}"
                for char in result.get("characters", [])
            ]
        )
        # 이미지 생성
        client = AsyncOpenAI(api_key=apikey)

        # 모든 이미지 생성 작업을 비동기로 실행
        tasks = [generate_cover_image(client, characters_info, result["book_cover_description"])] + [generate_page_image(client, characters_info, page, i) for i, page in enumerate(result["pages"])]
        # tqdm을 사용하여 진행 상황 표시
        print("이미지 생성 시작")
        all_results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating images"):
            all_results.append(await f)

        # 결과 처리
        result["title_img_path"] = all_results[0]  # 첫 번째 결과는 표지 이미지 URL
        
        # 페이지 결과만 필터링하고 정렬
        page_results = [r for r in all_results[1:] if isinstance(r, tuple) and len(r) == 2]
        result["pages"] = [page for _, page in sorted(page_results, key=lambda x: x[0])]
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"모든 이미지 생성 완료. 총 소요 시간: {total_time:.2f}초")
        print("모든 이미지 생성 완료")

        result.update({"user_id": user_id, "baby_id": baby_id})

        # JSON 파일로 결과 저장
        save_result_to_json(result, "fairytale_result4.json")
        print("JSON 파일 저장 완료")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

