from app.api.fairytale.utils.fairytale_utils import (
    load_diary_data,
    load_prompts,
    create_pairy_chain,
    save_result_to_json,
    view_generated_story,
)
from app.api.fairytale.utils.fairytale_image_utils import generate_image
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from app.api.fairytale.models import FairytaleInput
from tqdm import tqdm
import os

load_dotenv()

router = APIRouter()


@router.post("/generate_fairytale")
async def generate_fairytale(input_data: FairytaleInput):
    try:
        # 모델 인스턴스 데이터 -> 딕셔너리 데이터로 변환
        data = input_data.model_dump()

        # 프롬프트 로드
        prompts = load_prompts(
            "/Users/insu/pairy/Final-ml/app/api/fairytale/prompts/fairytale_prompt_ver1.txt"
        )

        # 동화 생성 체인 생성
        chain = create_pairy_chain(prompts, data)

        print("동화 생성 시작")
        # 동화 생성
        result = chain.invoke(
            {
                "name": data["name"],
                "emotion": data["emotion"],
                "activities": data["activities"],
                "special": data["special"],
            }
        )
        print("동화 생성 완료")

        # 이미지 생성
        dall_e = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for i in tqdm(
            range(len(result["pages"])),
            desc="이미지 생성 중",
            total=len(result["pages"]),
        ):
            image_url = generate_image(
                dall_e, result["pages"][i]["illustration_prompt"]
            )
            result["pages"][i]["image_url"] = image_url
        print("이미지 생성 완료")

        # JSON 파일로 결과 저장
        save_result_to_json(result, "fairytale_result.json")
        print("JSON 파일 저장 완료")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))