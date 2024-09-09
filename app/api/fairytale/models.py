from typing import List
from pydantic import BaseModel, Field


class StoryPage(BaseModel):
    """
    동화책의 한 페이지를 나타내는 클래스입니다.

    Attributes:
        text (str): 페이지의 텍스트 내용
        illustration_prompt (str): 이 페이지의 삽화를 생성하기 위한 프롬프트
    """

    text: str = Field(..., description="The text content of the story page")
    illustration_prompt: str = Field(
        ..., description="Prompt for generating an illustration for this page"
    )


class StoryResponse(BaseModel):
    """
    완성된 동화 이야기의 응답을 나타내는 클래스입니다.

    Attributes:
        title (str): 동화의 제목
        pages (List[StoryPage]): 동화의 페이지 목록
    """

    title: str = Field(..., description="The title of the story")
    pages: List[StoryPage] = Field(..., description="List of story pages")


class FairytaleInput(BaseModel):
    """
    동화 생성을 위한 입력 데이터를 나타내는 클래스입니다.

    Attributes:
        name (str): 주인공의 이름
        emotion (str): 주인공의 감정 상태
        health (str): 주인공의 건강 상태
        nutrition (str): 주인공의 영양 상태
        activities (List[str]): 주인공의 활동 목록
        social (str): 주인공의 사회적 상태
        special (str): 주인공의 특별한 특성이나 상황
        keywords (List[str]): 동화와 관련된 키워드 목록
    """

    user_id: int
    baby_id: int
    name: str
    emotion: str
    health: str
    nutrition: str
    activities: List[str]
    social: str
    special: str
    keywords: List[str]
