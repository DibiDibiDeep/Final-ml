from pydantic import BaseModel
from typing import List


# Dayinfo format
class DayInfoItem(BaseModel):
    user_id: int
    baby_id: int
    date: str
    role: str
    text: str

# embedd API Request format
class DayInfoBatch(BaseModel):
    items: List[DayInfoItem]