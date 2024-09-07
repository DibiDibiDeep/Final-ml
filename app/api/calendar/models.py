from pydantic import BaseModel
from langchain_core.pydantic_v1 import Field
from typing import List, Optional


class Activity(BaseModel):
    name: str
    time: Optional[str] = Field(default=None)
    infomation: str


class Event(BaseModel):
    date: str = Field(description="Day of the event (1-31)")
    activities: List[Activity]


class MonthlySchedule(BaseModel):
    year: Optional[str] = Field(default=None)
    month: str
    events: List[Event]
    etc: Optional[str] = Field(
        description="Additional information that does not fit into the event list",
        default=None,
    )
    user_id: str
    baby_id: str


# 이미지 업로드 모델(user_id, baby_id, image_path)
class ImageInput(BaseModel):
    user_id: str
    baby_id: str
    image_path: str
