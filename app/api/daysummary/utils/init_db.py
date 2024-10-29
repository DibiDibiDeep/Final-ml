from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    utility,
)
import os
from openai import OpenAI
from fastapi import APIRouter, HTTPException

# 환경변수 설정
openai_key = os.getenv("OPENAI_API_KEY")
milvus_host = os.getenv("MILVUS_HOST")
milvus_port = os.getenv("MILVUS_PORT")
embedding_model = os.getenv("EMBEDDING_MODEL")
embedding_dimension = os.getenv("EMBEDDING_DIMENSION")
collection_name = os.getenv("COLLECTION_NAME")

router = APIRouter()
# Milvus 연결 및 컬렉션 생성
connections.connect("default", host=milvus_host, port=str(milvus_port))

# Schema 생성
fields = [
    FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="baby_id", dtype=DataType.INT64),
    FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(
        name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimension
    ),
]

# Embedding Client 설정
client = OpenAI(api_key=openai_key)

# 컬렉션 schema 생성
schema = CollectionSchema(
    fields, "Collection for storing text and embeddings about child and parents"
)


# 컬렉션 생성
if utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' already exists. Skipping data insertion.")
    collection = Collection(collection_name)
else:
    collection = Collection(collection_name, schema)

    # 인덱스 생성
    collection.create_index(
        "embedding", {"index_type": "FLAT", "metric_type": "COSINE"}
    )

    # 백엔드 api에서 데이터 받아와서 동작.
    dummy = [
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-01",
            "role": "parents",
            "text": "뿌듯하고 감사해요 약간 피곤하지만 괜찮아요 아이와 함께 건강한 채소 위주의 식사를 했어요. 브로콜리 스프가 특히 맛있었어요! 아이 등하원, 업무, 저녁 산책 동료들과 협력하여 프로젝트를 무사히 마쳤어요. 저녁에는 이웃과 잠깐 대화를 나눴어요. 아이가 어제 배운 노래를 불러주었는데, 정말 감동이었어요. 아이의 성장을 눈으로 확인할 수 있어 행복했습니다.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-01",
            "role": "child",
            "text": "오늘은 정말 즐거운 하루였어! 😄\n아침에 친구들이랑 아이스크림 가게 역할놀이를 했어. 🍦\n나는 아이스크림을 팔고, 친구들은 손님이 되었지!\n아이스크림 먹는 연기를 정말 잘했어! 😋\n그 다음에는 놀이터에 가서 신나게 놀았어. 🛝\n미끄럼틀도 타고, 그네도 타고, 정말 재밌었어!\n친구들과 함께 웃음소리가 끊이지 않았어. 😂\n마지막으로 붓으로 그림을 그렸는데, 너무 즐거웠어! 🎨\n그림을 그리면서도 계속 웃고 있었어.\n오늘 하루가 너무 행복했어! 💖",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-02",
            "role": "parents",
            "text": "조금 지쳤지만 보람차요 허리가 약간 아파요 아침은 오트밀, 점심은 회사 구내식당, 저녁은 아이와 함께 삼계탕을 먹었어요. 아이 학교 준비물 챙기기, 업무 회의, 가족 저녁 식사 팀 회의에서 새로운 아이디어를 제안했어요. 저녁에는 가족과 오랜만에 대화의 시간을 가졌습니다. 아이가 학교에서 받아온 칭찬스티커를 보여줬는데, 정말 자랑스러웠어요.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-02",
            "role": "child",
            "text": "오늘은 정말 멋진 하루였어요! 💪\n체육 시간에 친구들이랑 릴레이 경기를 했는데, 우리 팀이 1등을 했어요! 🏃‍♂️🥇\n다 같이 힘을 합쳐서 뛰었더니 정말 뿌듯했어요.\n미술 시간에는 우리 가족 그림을 그렸는데, 선생님께서 정말 잘 그렸대요. 🎨👨‍👩‍👧\n방과 후에는 피아노 레슨도 갔어요. 새로운 곡을 배웠는데 조금 어려웠지만 열심히 연습할 거예요! 🎹\n오늘 하루는 정말 자신감이 넘치는 날이었어요. 내일도 이렇게 잘 할 수 있을 것 같아요! 😊",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-03",
            "role": "child",
            "text": "오늘은 정말 특별한 날이었어요! 🌟\n학교 학예회가 있었는데, 제가 노래 솔로 파트를 맡았어요. 🎤\n리허설 때는 너무 떨려서 실수도 했지만, 친구들이 응원해줘서 용기를 냈어요.\n실제 공연에서는 정말 잘 불렀어요! 부모님께서 눈물을 흘리시면서 박수를 쳐주셨어요. 😊\n공연 후에는 친구들과 작은 축하 파티를 했어요. 다들 서로 칭찬하고 축하해주는 게 정말 기분 좋았어요. 🎉\n비록 목이 좀 아프지만, 오늘은 제 인생에서 가장 자랑스러운 날 중 하나예요! 💖",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-04",
            "role": "parents",
            "text": "걱정되지만 희망적이에요 감기 기운이 있어요 따뜻한 국물 위주로 식사했어요. 저녁엔 아이와 함께 건강한 된장찌개를 끓였어요. 재택근무, 아이 숙제 도와주기, 병원 방문 화상 회의로 팀원들과 소통했어요. 아이의 담임 선생님과 전화 상담을 했습니다. 아이가 처음으로 혼자 단추를 채웠어요. 작지만 큰 성장을 느꼈습니다.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-04",
            "role": "child",
            "text": "오늘은 조금 이상한 하루였어요. 🤒\n학교에 가지 않고 집에서 온라인으로 수업을 들었어요. 선생님 얼굴을 화면으로 보는 게 신기했어요.\n기침이 나고 몸이 안 좋아서 공부하기가 조금 힘들었지만, 엄마가 계속 옆에서 도와주셨어요. 💖\n점심 먹고 나서는 새로 산 공룡 퍼즐을 맞췄어요. 어려웠지만 재미있었어요! 🦕\n저녁에는 혼자서 옷 단추를 다 채웠어요. 엄마가 정말 기뻐하시던게 기억나요. 😊\n아픈 친구한테 전화해서 괜찮은지 물어봤어요. 내일은 우리 둘 다 나아있기를 바라요. 🙏",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-06",
            "role": "parents",
            "text": "자부심과 기대감이 넘쳐요 활기차고 건강해요 아침은 통곡물 샌드위치, 점심은 회사 구내식당에서 한식, 저녁은 아이와 함께 만든 피자를 먹었어요. 출근, 프레젠테이션, 요리 시간, 취침 전 독서 중요한 프레젠테이션에서 좋은 평가를 받았어요. 저녁에는 아이와 함께 요리하며 대화를 나눴습니다. 아이가 처음으로 피자 도우를 직접 만들었어요. 함께 요리하는 시간이 정말 특별했습니다.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-06",
            "role": "child",
            "text": "오늘은 정말 재미있는 하루였어요! 😃\n학교에서 과학 시간에 멋진 실험을 했어요. 친구들이랑 같이 하니까 더 재미있었어요. 🧪\n집에 와서는 엄마 아빠랑 같이 피자를 만들었어요. 제가 처음으로 피자 도우를 만들어봤는데, 조금 어려웠지만 정말 재미있었어요! 🍕\n제가 만든 피자를 가족들이 맛있게 먹어줘서 너무 뿌듯했어요. \n저녁에는 새로 산 책을 읽었어요. 모험 이야기라 정말 재미있었어요! 📚\n오늘 하루는 정말 특별하고 즐거웠어요. 내일도 이렇게 재미있는 일이 있으면 좋겠어요! 💖",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-12",
            "role": "parents",
            "text": "조금 피곤하지만 만족스러워요 허리가 약간 아프지만 괜찮아요 아침은 오트밀, 점심은 동료들과 외식, 저녁은 간단한 샐러드를 먹었어요. 출근, 프로젝트 미팅, 아이 숙제 도와주기, 집안일 점심에 동료들과 즐거운 대화를 나눴어요. 저녁에는 아이의 학교생활에 대해 이야기를 들었습니다. 아이가 어려운 수학 문제를 스스로 해결했어요. 아이의 성장을 느낄 수 있어 뿌듯했습니다.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-12",
            "role": "child",
            "text": "오늘은 정말 뿌듯한 하루였어요! 😊\n수학 시간에 어려운 문제가 나왔는데, 열심히 생각해서 혼자 힘으로 풀었어요. 선생님께서 정말 잘했다고 칭찬해주셨어요! 🎉\n체육 시간에는 친구들이랑 농구를 했는데, 제가 결승점을 넣었어요! 팀원들이 정말 기뻐했어요. 🏀\n점심으로 나온 비빔밥이 정말 맛있었어요. 야채도 많이 먹었답니다. 🥗\n방과 후에는 미술 동아리에 갔는데, 거기서 새로운 친구도 사귀었어요. 함께 그림 그리는 게 정말 재미있었어요. 🎨\n집에 와서 숙제를 하는데, 엄마가 도와주셔서 금방 끝냈어요.\n오늘 하루는 정말 행복하고 자신감이 넘쳤어요. 내일도 이렇게 좋은 일만 가득하면 좋겠어요! 💖",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-18",
            "role": "parents",
            "text": "약간 긴장되지만 희망적이에요 목에 약간 통증이 있어요 아침은 과일 스무디, 점심은 회사 구내식당, 저녁은 아이와 함께 만든 김치찌개를 먹었어요. 중요 회의 준비, 프레젠테이션, 장보기, 요리 중요한 회의에서 새로운 아이디어를 제안했어요. 저녁에는 아이와 함께 요리하며 대화를 나눴습니다. 아이가 처음으로 김치찌개 만드는 것을 도와줬어요. 함께 요리하는 모습이 정말 대견했습니다.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-18",
            "role": "child",
            "text": "오늘은 정말 재미있고 새로운 경험을 많이 한 날이에요! 😃\n학교에서 과학 시간에 신기한 실험을 했어요. 물의 표면장력에 대해 배웠는데, 정말 놀라웠어요! 🧪\n점심 시간에는 도서관에 가서 새로운 책을 빌렸어요. 공룡에 대한 책인데 정말 재미있어 보여요. 📚\n학교가 끝나고 엄마 아빠랑 같이 장을 보러 갔어요. 제가 직접 장바구니를 들고 다니면서 필요한 재료를 골랐어요. 🛒\n집에 와서는 엄마 아빠랑 함께 김치찌개를 만들었어요. 제가 처음으로 김치를 썰어봤는데, 조금 어려웠지만 재미있었어요! 🥘\n우리가 함께 만든 김치찌개는 정말 맛있었어요. 요리하는 게 이렇게 재미있는 줄 몰랐어요.\n오늘 하루는 정말 특별하고 즐거웠어요. 내일은 또 어떤 새로운 것을 배울 수 있을지 기대돼요! 💖",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-24",
            "role": "parents",
            "text": "뿌듯하고 감사해요 컨디션이 좋아요 아침은 통곡물 시리얼, 점심은 동료와 샐러드, 저녁은 가족과 함께 삼겹살을 구워 먹었어요. 업무 마무리, 동료와 점심, 가족 저녁 식사, 아이와 산책 점심에 동료와 중요한 프로젝트 성공을 축하했어요. 저녁에는 가족과 오랜만에 외식을 즐겼습니다. 아이가 학교에서 받은 '이 달의 모범학생' 상장을 보여줬어요. 정말 자랑스러웠습니다.",
        },
        {
            "user_id": 1,
            "baby_id": 1,
            "date": "2024-09-24",
            "role": "child",
            "text": "오늘은 정말 특별하고 행복한 날이었어요! 😊\n학교에서 '이 달의 모범학생' 상을 받았어요. 선생님께서 제 이름을 부르셨을 때 정말 놀랐어요! 🏆\n친구들이 축하해줘서 정말 기분이 좋았어요. \n방과 후에는 엄마 아빠가 축하한다고 맛있는 삼겹살을 먹으러 갔어요. 오랜만에 외식을 해서 정말 즐거웠어요. 🥓\n저녁 식사 후에는 가족과 함께 동네를 산책했어요. 날씨도 좋고 정말 행복한 시간이었어요. 🌙\n오늘 하루는 정말 자랑스럽고 행복했어요. 앞으로도 열심히 해서 더 자주 칭찬받고 싶어요! 💖",
        },
    ]

    def get_embedding(client, text, model= embedding_model):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    # 데이터 준비 및 삽입
    entities = []
    for item in dummy:
        text = item["text"]
        entity = {
            "user_id": item["user_id"],
            "baby_id": item["baby_id"],
            "date": item["date"],
            "role": item["role"],
            "text": text,
            "embedding": get_embedding(client, text),
        }
        entities.append(entity)
    # 데이터 삽입
    collection.insert(entities)
    print(f"Inserted {len(entities)} entities into the collection")
# 컬렉션 로드
collection.load()
# 컬렉션 플러시
collection.flush()
