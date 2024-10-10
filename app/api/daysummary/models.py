from pydantic import BaseModel
from typing import Dict, List
from uuid import uuid4

# 사용자 쿼리를 처리하는 클래스
class Query(BaseModel):
    baby_id: int = 1
    user_id: int = 1
    session_id: str = None
    text: str

#  AI 채팅 시스템의 대화 기록을 관리하는 클래스
class AIChatHistory:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, List[str]]] = {}

    # 사용자 ID와 세션 ID를 조합하여 고유한 세션 키를 생성
    def get_session_key(self, user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"
    
    # 세션 ID가 없으면 새로 생성
    def get_or_create_session(self, user_id: str, session_id: str = None) -> str:
        if not session_id:
            session_id = str(uuid4())
        return session_id
    
    # 특정 사용자, 세션, 아기에 대한 대화 기록 반환
    def get_chat_history(self, user_id: str, session_id: str, baby_id: str) -> List[str]:
        session_key = self.get_session_key(user_id, session_id)
        return self.sessions.setdefault(session_key, {}).get(baby_id, [])
    
    # 새로운 메시지를 대화 기록에 추가. 사용자와 AI의 메시지를 구분.(is_user)
    def add_message(self, user_id: str, session_id: str, baby_id: str, message: str, is_user: bool = True):
        session_key = self.get_session_key(user_id, session_id)
        if session_key not in self.sessions:
            self.sessions[session_key] = {}
        if baby_id not in self.sessions[session_key]:
            self.sessions[session_key][baby_id] = []
        
        prefix = "User: " if is_user else "Bot: "
        self.sessions[session_key][baby_id].append(f"{prefix}{message}")

    #  특정 대화 기록을 초기화
    def reset_history(self, user_id: str, session_id: str, baby_id: str):
        session_key = self.get_session_key(user_id, session_id)
        if session_key in self.sessions and baby_id in self.sessions[session_key]:
            self.sessions[session_key][baby_id] = []

    # 전체 대화 기록을 문자열로 반환
    def get_full_history(self, user_id: str, session_id: str, baby_id: str) -> str:
        return str(self.get_chat_history(user_id, session_id, baby_id))