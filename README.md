# Final-ml

## 사용법

### 환경설정 순서
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

#### calendar
```bash
# request
{
  "image_path": "images/sch8.jpg"
}

# response
{
  "events": [
    {
      "date": "06",
      "activities": [
        "물놀이 (4세)",
        "열린어린이집 바다반 활동보조",
        "빨간망토 인형극 공연 관람 [비대면] (3세~5세)",
        "생일 축하: 김주원, 내민율, 이시원"
      ]
    },
    {
      "date": "07",
      "activities": [
        "소방대피훈련"
      ]
    },
    {
      "date": "08",
      "activities": [
        "지역사회연계활동 - 마트 (4세, 5세)",
        "열린어린이집 바다반 활동보조"
      ]
    }
    ...
  ],
  "etc": "국공립 천사어린이집"
}
```

#### babydiary
```bash
# request

어머니~ XXX이 멋진 우비를 입고 짠~^^하고 등장했네요 ㅎㅎ 오늘도 역시나 자동차를 가지고 왔네요 빨간차가지고 다투니까 아예 빨간 차로만 가져왔네요ㅎ 친구들 골고루나눠주고 잘 놀았어요 시소 끼적이기, 도장찍기도 하면서 즐겁게 잘지냈습니다~^^ 오전간식 요플레, 점심도 김가루하고 야무지게 먹고 양치하고 잠자리에 들었어요 기침을 간혹 하네요 집에서도 잘 관찰해 주세요.

# response
{
  "name": "XXX",
  "emotion": "즐거움과 행복",
  "health": "가끔 기침을 함, 집에서도 잘 관찰해 주세요.",
  "nutrition": "오전 간식으로 요플레를 먹고, 점심으로 김가루를 잘 먹었어요.",
  "activities": [
    "자동차 가지고 놀기",
    "시소 타기",
    "도장 찍기"
  ],
  "social": "친구들과 골고루 나눠주며 잘 놀았어요.",
  "special": "멋진 우비를 입고 등장했어요.",
  "keywords": [
    "우비",
    "자동차",
    "요플레",
    "김가루",
    "기침"
  ],
  "diary": "오늘은 정말 즐거운 하루였어요! 😊  \n아침에 멋진 우비를 입고 나갔어요. 🌧️  \n오전 간식으로 요플레를 먹었는데, 정말 맛있었어요! 🍦  \n점심에는 김가루를 잘 먹었어요. 맛있었어요! 🍚  \n\n그 후에 자동차 가지고 놀았어요. 🚗  \n친구들과 함께 시소도 탔어요. 너무 재밌었어요! 🎠  \n그리고 도장 찍기도 했어요. 예쁜 도장이 많이 나왔어요! 🌟  \n\n친구들과 골고루 나눠주며 잘 놀았어요. 🤗  \n가끔 기침을 했지만, 엄마가 잘 지켜봐 주셨어요. ❤️  \n오늘 하루가 정말 행복했어요! 🌈"
}
```