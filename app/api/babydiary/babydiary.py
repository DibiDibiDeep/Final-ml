from dotenv import load_dotenv

import json, os
from fastapi import APIRouter, HTTPException

from app.api.babydiary.models import DiaryInput
from app.api.babydiary.utils.chain_util import (
    setup_extract_keyword_chain,
    setup_write_diary_chain,
)

load_dotenv()  # local .env에서 불러옴


extract_keyword_chain = setup_extract_keyword_chain()
write_diary_chain = setup_write_diary_chain()

router = APIRouter()


@router.post("/generate_diary")
async def process_report(diary_input: DiaryInput):

    # try:
    user_id = diary_input.user_id
    baby_id = diary_input.baby_id
    notice = diary_input.report

    # 키워드 추출 체인 실행
    report = extract_keyword_chain.invoke({"report": notice})

    # 유효한 일기가 아니면 예외 발생
    if report["is_valid"] == False:
        raise HTTPException(status_code=400, detail="Invalid daycare report content")
    # 유효한 경우 검증 키값 제거
    report.pop("is_valid", None)

    # 일기 작성 체인 실행
    result = write_diary_chain.invoke(
        {
            "name": report["name"],
            "emotion": report["emotion"],
            "health": report["health"],
            "nutrition": report["nutrition"],
            "activities": ", ".join(report["activities"]),
            "social": report["social"],
            "special": report["special"],
        }
    )
    report["diary"] = result
    report.update({"user_id": user_id, "baby_id": baby_id, "role": "child"})

    return report
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
