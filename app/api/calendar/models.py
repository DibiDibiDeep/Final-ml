from pydantic import BaseModel, Field
from typing import List, Optional


class Activity(BaseModel):
    name: str = Field(..., description="Name of the activity")
    start_time: Optional[str] = Field(
        None, description="Start time of the activity (e.g. 09:30:00)"
    )
    end_time: Optional[str] = Field(
        None, description="End time of the activity (e.g. 10:30:00)"
    )
    location: str = Field(default="원내", description="Location of the activity")
    # target: Optional[str] = Field(
    #     default="전체", description="Target group for the activity (e.g. 꽃잎반)"
    # )
    target: str = Field(
        default="전체", description="Target group for the activity (e.g. 꽃잎반)"
    )
    information: str = Field(
        ..., description="Main information or content of the activity"
    )
    notes: Optional[str] = Field(None, description="Additional notes or description")


class Event(BaseModel):
    date: str = Field(..., description="Day of the event (1-31)")
    activities: List[Activity] = Field(
        default_factory=list, description="List of activities for the event"
    )


class MonthlySchedule(BaseModel):
    year: Optional[str] = Field(None, description="Year of the schedule")
    month: str = Field(..., description="Month of the schedule")
    events: List[Event] = Field(
        default_factory=list, description="List of events in the month"
    )
    etc: Optional[str] = Field(
        None, description="Additional information that does not fit into the event list"
    )


# 이미지 업로드 모델(user_id, baby_id, image_path)
class ImageInput(BaseModel):
    user_id: int = Field(..., description="User ID")
    baby_id: int = Field(..., description="Baby ID")
    image_path: str = Field(..., description="Path to the uploaded image")
