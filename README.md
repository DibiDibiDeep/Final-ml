# 몽글몽글 - ML Repository

## 역할
|김인수 |옥창우
|:-:|:-:|
|<img src='https://github.com/user-attachments/assets/eb4d8fb3-4437-4fc0-8f00-d2480f37bca7' height=160 width=125></img>|<img src='https://github.com/user-attachments/assets/348a33f6-0b40-4d61-8108-c26f985f13d6' height=160 width=125></img>|<img src='https://github.com/user-attachments/assets/348a33f6-0b40-4d61-8108-c26f985f13d6' height=160 width=125></img>|
|- AI 서비스 개발<br>(가정통신문, 알림장, 동화, 챗봇)<br> - 프롬프트 엔지니어링<br>- Docker 컨테이너화<br>- API 서버 개발|- AWS 배포<br>- 개발 환경 표준화<br>- Docker 컨테이너화<br>- 모니터링 및 이슈파악

## API 디렉토리 구조
```
├── api
│   ├── audiomemo  # 음성메모
│   ├── babydiary  # 알림장 분석
│   ├── calendar   # 가정통신문 분석
│   ├── daysummary # 하루요약 챗봇
│   ├── embedding  # 벡터DB 임베딩
│   └── fairytale  # 동화 생성
└── main.py
```
## 사용법

- root디렉토리에 `.env`파일 추가
```bash
OPENAI_API_KEY=your-api-key

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-api-key
LANGCHAIN_PROJECT=your-project-name

AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_KEY=your-secret-access-key
AWS_REGION=your-region
AWS_S3_BUCKET=your-bucket-name
```

- docker image build
```bash
# build image
docker build -t [IMAGENAME]:[TAG] .

# run container
docker run --env-file .env --name [CONTAINERNAME] -p 8000:8000 [IMAGENAME]:[TAG] 
```

- 이후 호스트ip:8000으로 접근해서 스웨거에 값 입력으로 테스트 가능


### 로컬에서 실행 시 환경설정
```bash
conda create -n [ENVNAME] python=3.11.0
conda activate [ENVNAME]
pip install -r requirements.txt
conda install -c conda-forge tesseract
```


#### If Windows:
```bash
pip install uvicorn[standard]
```

### 실행 예시
- `python app/main.py` 또는 `uvicorn app.main:app --reload`

#### calendar
- http://127.0.0.1:8000/docs 에서 image_path 입력 후 실행 또는 아래 명령어 새로운 터미널에서 실행.
    - result 디렉토리 없으면 생성. 모든 과정이 끝나면 해당 디렉토리안에 결과물 Json 파일 저장.
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/process_image' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"image_path": "images/sch8.jpg"}'
```

#### babydiary
- http://127.0.0.1:8000/docs 에서 텍스트 입력 후 실행 또는 아래 명령어 새로운 터미널에서 실행.
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate_diary?input_notice=[your input text]' \
  -H 'accept: application/json' \
  -d ''
```

### 실행 결과

# calendar
## request
<img src="https://github.com/user-attachments/assets/17b5f928-be0b-4028-a85a-621f10140bb3" width="500" height="600"/>

```bash
{
  "user_id": 0,
  "baby_id": 0,
  "image_path": "images/sch8.jpg" # or s3 bucket url
}
```
## response
```bash
{
  "year": null,
  "month": "04",
  "events": [
    {
      "date": "05",
      "activities": [
        {
          "name": "식목일 행사",
          "start_time": null,
          "end_time": null,
          "location": "원내",
          "target": "전체",
          "information": "식목일 행사",
          "notes": null
        },
        {
          "name": "지진대피훈련",
          "start_time": "09:50:00",
          "end_time": "10:00:00",
          "location": "원내",
          "target": "전체",
          "information": "지진대피훈련",
          "notes": "9시30분 등원시간을 지켜주세요."
        }
      ]
    },
    {
      "date": "06",
      "activities": [
        {
          "name": "동지 후 105일째 되는 날",
          "start_time": null,
          "end_time": null,
          "location": "원내",
          "target": "전체",
          "information": "동지 후 105일째 되는 날",
          "notes": null
        }
      ]
    },
    {
      "date": "19",
      "activities": [
        {
          "name": "생일잔치",
          "start_time": "10:00:00",
          "end_time": "11:00:00",
          "location": "원내",
          "target": "전체",
          "information": "내가 태어났어요!(생일잔치)",
          "notes": "많이 축하해주세요."
        }
      ]
    },
    {
      "date": "26",
      "activities": [
        {
          "name": "정기 소방안전교육 및 대피훈련",
          "start_time": "09:50:00",
          "end_time": "10:00:00",
          "location": "원내",
          "target": "전체",
          "information": "정기 소방안전교육 및 대피훈련",
          "notes": null
        }
      ]
    }
  ],
  "etc": "학부모 개별상담 안내: 4월 10일부터 어린이집 생활과 성장발달에 관한 개별상담이 진행됩니다. 상담 날짜와 시간은 반별로 조율됩니다. \n4월 특별활동비 안내: 체육(만2~5세) 13,000원, 음악(만2~5세) 14,000원, 영어(만3~5세) 38,700원. \n야간연장 및 토요보육 신청은 사무실로 사전 신청 바랍니다. 야간연장 보육은 신청서 제출 필요.",
  "user_id": 0,
  "baby_id": 0
}
}

```

# babydiary
## request
```bash
{
  "user_id": 0,
  "baby_id": 0,
  "report": "어머니~ 우현이 멋진 우비를 입고 짠~^^하고 등장했네요 ㅎㅎ 오늘도 역시나 자동차를 가지고 왔네요 빨간차가지고 다투니까 아예 빨간 차로만 가져왔네요ㅎ 친구들 골고루나눠주고 잘 놀았어요 시소 끼적이기, 도장찍기도 하면서 즐겁게 잘지냈습니다~^^ 오전간식 요플레, 점심도 김가루하고 야무지게 먹고 양치하고 잠자리에 들었어요 기침을 간혹 하네요 집에서도 잘 관찰해 주세요."
}
```

## response
```bash
{
  "name": "우현",
  "emotion": "즐거움과 행복",
  "health": "기침을 간혹 함, 집에서도 잘 관찰 필요",
  "nutrition": "오전 간식으로 요플레를 먹고, 점심으로 김가루를 잘 먹음",
  "activities": [
    "자동차 가지고 놀기",
    "시소 타기",
    "도장 찍기"
  ],
  "social": "친구들과 골고루 나누며 잘 놀았음",
  "special": "멋진 우비를 입고 등장함",
  "keywords": [
    "우비",
    "자동차",
    "요플레",
    "김가루",
    "기침"
  ],
  "diary": "오늘은 정말 즐거운 하루였어! 😊  \n아침에 일어나서 멋진 우비를 입고 나갔어. 🌧️  \n친구들이랑 자동차 가지고 놀았어. 🚗  \n우리는 시소도 타고, 정말 신났어! 🎉  \n도장 찍기도 했는데, 너무 재밌었어! 🖌️  \n점심으로 김가루를 잘 먹었고, 오전 간식으로 요플레도 먹었어. 🍦  \n가끔 기침을 했지만, 엄마가 잘 지켜봐 주셨어.  \n친구들과 골고루 나누며 잘 놀았고, 모두 행복했어! 😄  \n오늘 하루가 정말 즐거웠어! 🌈",
  "user_id": 0,
  "baby_id": 0,
  "role": "child"
}
```

# daysummary
## request
```bash
{
  "user_id": 0,
  "baby_id": 0,
  "session_id": "session-id",
  "text": "내 오늘 하루는 어땠지?"
}
```
## response
```bash
{
  "user_id": 0,
  "baby_id": 0,
  "session_id": "your-id",
  "response": "답변 - ",
}
```

# fairytale
## request
```bash
{
  "name": "지수",
  "emotion": "즐거움과 신남",
  "health": "활기차고 건강함",
  "nutrition": "식사에 대한 정보는 제공되지 않음",
  "activities": [
    "아이스크림 가게 역할놀이",
    "놀이터에서 놀기",
    "붓으로 그림 그리기"
  ],
  "social": "친구들과 함께 즐겁게 놀며 웃음소리를 나누었음",
  "special": "아이스크림 가게 역할을 진지하게 맡아 연기함",
  "keywords": [
    "아이스크림",
    "역할놀이",
    "놀이터",
    "그림",
    "웃음",
    "잠자리"
  ],
  "diary": "오늘은 정말 신나는 하루였어! 😄  \n아침에 일어나서 기분이 너무 좋았어. ☀️  \n나는 친구들과 함께 아이스크림 가게 역할놀이를 했어. 🍦  \n내가 아이스크림 가게 주인 역할을 맡았는데, 진짜로 가게를 운영하는 것처럼 연기했어! 🎭  \n친구들이 와서 다양한 맛의 아이스크림을 주문했어.  \n우리는 서로 웃으면서 즐거운 시간을 보냈어! 😂  \n\n그 다음에는 놀이터로 갔어. 🛝  \n미끄럼틀도 타고, 그네도 타고, 정말 신났어!  \n친구들과 함께 뛰어놀면서 웃음소리가 끊이지 않았어. 🎉  \n놀이터에서의 시간은 항상 너무 재밌어!  \n\n마지막으로 붓으로 그림을 그렸어. 🎨  \n색깔이 너무 예쁘고, 내 그림이 멋지게 나왔어!  \n오늘 하루는 정말 즐거웠고, 나는 건강하고 활기차! 💪  \n내일도 이렇게 신나는 하루가 되길 바래! 🌈",
  "user_id": 0,
  "baby_id": 0,
  "role": "child"
}
```

## response
```bash
{
  "title": "지수의 마법 아이스크림 모험 🍦✨",
  "pages": [
    {
      "text": "오늘은 지수가 아이스크림 가게 역할놀이를 하기로 했어요! 지수는 아이스크림 가게의 주인으로 변신했답니다. '어서 오세요! 어떤 아이스크림을 원하세요?'라고 외치며, 친구들을 맞이했어요. 🍨",
      "illustration_prompt": "A cheerful child, 지수, wearing an apron and a hat, standing behind a colorful ice cream counter with various ice cream flavors and toppings.",
      "illustration": "base64_incoding_value"
    },
    {
      "text": "지수는 친구들에게 다양한 맛의 아이스크림을 만들어 주었어요. '이건 딸기 아이스크림이에요! 그리고 이건 초코 아이스크림!' 친구들은 신나서 아이스크림을 먹으며 즐거운 시간을 보냈어요. 그런데 갑자기, 아이스크림 가게의 문이 열리더니, 마법의 요정이 나타났어요! 🧚‍♀️",
      "illustration_prompt": "A magical fairy with sparkling wings appearing in an ice cream shop, surrounded by colorful ice cream cones and happy children.",
      "illustration": "base64_incoding_value"
    },
    {
      "text": "요정은 지수에게 말했어요. '지수야, 너의 아이스크림 가게는 마법의 힘을 가지고 있어! 아이스크림을 먹으면 놀이터로 순간 이동할 수 있어!' 지수는 놀라서 '정말요?'라고 물었어요. 요정은 고개를 끄덕이며, 지수와 친구들에게 마법의 아이스크림을 주었어요. 🍭",
      "illustration_prompt": "The fairy handing out magical ice cream cones to 지수 and her friends, with sparkles and a magical aura around them.",
      "illustration": "base64_incoding_value"
    },
    {
      "text": "지수와 친구들은 마법의 아이스크림을 한 입 먹자, 눈 깜짝할 사이에 놀이터로 이동했어요! 놀이터는 환상적인 색깔로 가득 차 있었고, 신나는 놀이기구들이 가득했어요. '와! 신난다!' 지수는 신나서 미끄럼틀을 타고, 그네를 탔어요. 🎠",
      "illustration_prompt": "A vibrant playground filled with colorful slides and swings, with 지수 and her friends joyfully playing.",
      "illustration": "base64_incoding_value"
    },
    {
      "text": "놀이터에서 신나게 놀다가, 지수는 붓과 물감을 발견했어요. '이걸로 그림을 그려볼까?' 지수는 친구들과 함께 멋진 그림을 그리기 시작했어요. 그들은 하늘을 날고 있는 아이스크림과 마법의 요정을 그렸답니다. 🎨",
      "illustration_prompt": "지수 and her friends painting a colorful mural of flying ice cream and a magical fairy, with paint splatters around them.",
      "illustration": "base64_incoding_value"
    },
    {
      "text": "그림을 다 그리고 나니, 지수는 오늘의 모험이 정말 특별하다는 것을 느꼈어요. '오늘은 정말 즐거운 날이었어!' 지수는 친구들과 함께 웃으며, 마법의 아이스크림 가게로 돌아갔답니다. 그리고 그들은 다시 만날 것을 약속했어요. 🌈",
      "illustration_prompt": "지수 and her friends happily walking back to the ice cream shop, with smiles on their faces and colorful ice cream cones in their hands.",
      "illustration": "base64_incoding_value"
    }
  ],
  "cover_illustration": "base64_incoding_value",
  "user_id": 0,
  "baby_id": 0
}
```
