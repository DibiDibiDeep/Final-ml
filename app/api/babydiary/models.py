from pydantic import BaseModel, Field
from typing import List, Optional


# 키워드 추출 모델
class DaycareReport(BaseModel):
    is_valid: bool = Field(..., description="Whether the report is valid")
    name: Optional[str] = Field(None, description="Child's name")
    emotion: Optional[str] = Field(
        None, description="Child's overall mood and emotional state"
    )
    health: Optional[str] = Field(
        None, description="Child's physical health and well-being"
    )
    nutrition: Optional[str] = Field(
        None, description="Summary of child's meals and eating behavior"
    )
    activities: List[str] = Field(description="Main activities the child engaged in")
    social: Optional[str] = Field(
        None, description="Child's interactions with peers and teachers"
    )
    special: str = Field(description="Special achievements or unusual occurrences")
    keywords: Optional[List[str]] = Field(
        None, description="Important keywords from entities other than name entities"
    )


# 일기 작성 모델
class DiaryInput(BaseModel):
    user_id: int
    baby_id: int
    report: str
