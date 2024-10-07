from pydantic import BaseModel


class Query(BaseModel):
    baby_id: int = 1
    user_id: int = 1
    session_id: str = None
    text: str
