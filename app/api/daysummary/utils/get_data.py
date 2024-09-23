# 오늘의 부모, 아이의 하루 정보 추출 과정을 거쳐서 추출되었다고 가정.


today_child_events = {
    "date": "2024-09-04",
    "role": "child",
    "emotion": "궁금하고 걱정돼요",
    "health": "기침이 조금 나요",
    "nutrition": "입맛이 없어서 평소보다 적게 먹었어요.",
    "activities": ["온라인 수업", "책 읽기", "퍼즐 맞추기"],
    "social": "아픈 친구에게 영상통화로 안부를 물었어요.",
    "special": "혼자서 옷 단추를 다 채웠어요! 엄마가 정말 기뻐하셨어요.",
    "keywords": ["온라인수업", "책", "퍼즐", "단추", "영상통화"],
    "text": "오늘은 조금 이상한 하루였어요. 🤒\n학교에 가지 않고 집에서 온라인으로 수업을 들었어요. 선생님 얼굴을 화면으로 보는 게 신기했어요.\n기침이 나고 몸이 안 좋아서 공부하기가 조금 힘들었지만, 엄마가 계속 옆에서 도와주셨어요. 💖\n점심 먹고 나서는 새로 산 공룡 퍼즐을 맞췄어요. 어려웠지만 재미있었어요! 🦕\n저녁에는 혼자서 옷 단추를 다 채웠어요. 엄마가 정말 기뻐하시던게 기억나요. 😊\n아픈 친구한테 전화해서 괜찮은지 물어봤어요. 내일은 우리 둘 다 나아있기를 바라요. 🙏",
}

today_parent_events = {
    "date": "2024-09-04",
    "role": "parents",
    "emotion": "걱정되지만 희망적이에요",
    "health": "감기 기운이 있어요",
    "nutrition": "따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요.",
    "activities": ["재택근무", "아이 숙제 도와주기", "병원 방문"],
    "social": "화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다.",
    "special": "아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
    "keywords": ["재택근무", "숙제", "병원", "화상회의", "성장"],
    "text": "걱정되지만 희망적이에요 감기 기운이 있어요 따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요. 재택근무, 아이 숙제 도와주기, 병원 방문 화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다. 아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
}

today_text = f"오늘 날짜: {today_child_events['date']}\n아이의 하루: {today_child_events['text']}\n부모의 하루: {today_parent_events['text']}"
today_info = {"today_text": today_text}
