from pydantic import BaseModel
from langchain_core.pydantic_v1 import Field
from typing import List


class Event(BaseModel):
    date: str = Field(description="Day of the event datetime format DD(1-31)")
    activities: List[str] = Field(
        description="Concise list of activities for this date, including only essential information"
    )


class MonthlySchedule(BaseModel):
    events: List[Event] = Field(description="List of events for each date")
    etc: str = Field(
        description="Additional information that does not fit into the event list"
    )


class ImageInput(BaseModel):
    image_path: str
