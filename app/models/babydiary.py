from pydantic import BaseModel, Field
from typing import List


class DaycareReport(BaseModel):
    name: str = Field(description="Child's name")
    emotion: str = Field(description="Child's overall mood and emotional state")
    health: str = Field(description="Child's physical health and well-being")
    nutrition: str = Field(description="Summary of child's meals and eating behavior")
    activities: List[str] = Field(description="Main activities the child engaged in")
    social: str = Field(description="Child's interactions with peers and teachers")
    special: str = Field(description="Special achievements or unusual occurrences")
    keywords: List[str] = Field(
        description="Important keywords from entities other than name entities"
    )
