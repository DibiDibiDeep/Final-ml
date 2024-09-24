from datetime import datetime
# 오늘의 부모, 아이의 하루 정보 추출 과정을 거쳐서 추출되었다고 가정.

def get_today_info(user_id: int, baby_id: int):
    today_child_events = {
        "user_id": 1,
        "baby_id": 1,
        "date": "2024-09-24",
        "role": "child",
        # "emotion": "궁금하고 걱정돼요",
        # "health": "기침이 조금 나요",
        # "nutrition": "입맛이 없어서 평소보다 적게 먹었어요.",
        # "activities": ["온라인 수업", "책 읽기", "퍼즐 맞추기"],
        # "social": "아픈 친구에게 영상통화로 안부를 물었어요.",
        # "special": "혼자서 옷 단추를 다 채웠어요! 엄마가 정말 기뻐하셨어요.",
        # "keywords": ["온라인수업", "책", "퍼즐", "단추", "영상통화"],
        "text": "오늘은 정말 특별하고 행복한 날이었어요! 😊\n학교에서 '이 달의 모범학생' 상을 받았어요. 선생님께서 제 이름을 부르셨을 때 정말 놀랐어요! 🏆\n친구들이 축하해줘서 정말 기분이 좋았어요. \n방과 후에는 엄마 아빠가 축하한다고 맛있는 삼겹살을 먹으러 갔어요. 오랜만에 외식을 해서 정말 즐거웠어요. 🥓\n저녁 식사 후에는 가족과 함께 동네를 산책했어요. 날씨도 좋고 정말 행복한 시간이었어요. 🌙\n오늘 하루는 정말 자랑스럽고 행복했어요. 앞으로도 열심히 해서 더 자주 칭찬받고 싶어요! 💖",
        }

    today_parent_events = {
        "user_id": 1,
        "baby_id": 1,
        "date": "2024-09-24",
        "role": "parents",
        # "emotion": "걱정되지만 희망적이에요",
        # "health": "감기 기운이 있어요",
        # "nutrition": "따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요.",
        # "activities": ["재택근무", "아이 숙제 도와주기", "병원 방문"],
        # "social": "화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다.",
        # "special": "아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
        # "keywords": ["재택근무", "숙제", "병원", "화상회의", "성장"],
        "text": "뿌듯하고 감사해요 컨디션이 좋아요 아침은 통곡물 시리얼, 점심은 동료와 샐러드, 저녁은 가족과 함께 삼겹살을 구워 먹었어요. 업무 마무리, 동료와 점심, 가족 저녁 식사, 아이와 산책 점심에 동료와 중요한 프로젝트 성공을 축하했어요. 저녁에는 가족과 오랜만에 외식을 즐겼습니다. 아이가 학교에서 받은 '이 달의 모범학생' 상장을 보여줬어요. 정말 자랑스러웠습니다.",
        }
    today_date = datetime.now().strftime("%Y-%m-%d")
    if today_child_events['user_id'] == user_id and today_child_events['baby_id'] == baby_id and today_parent_events['user_id'] == user_id and today_parent_events['baby_id'] == baby_id:
        if today_child_events['date'] == today_date and today_parent_events['date'] == today_date:
            today_text = f"\nToday Date: {today_child_events['date']}\nChild Day Info: {today_child_events['text']}\nParents Day Info: {today_parent_events['text']}"
            today_info = {"today_text": today_text}
    else:
        today_info = {"today_text": "No data"}
    return today_info
