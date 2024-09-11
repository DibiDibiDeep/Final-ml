# Final-ml

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
```bash
{
  "user_id": "string",
  "baby_id": "string",
  "image_path": "images/sch8.jpg"
}
```
## response
```bash
{
  "year": null,
  "month": "08",
  "events": [
    {
      "date": "06",
      "activities": [
        {
          "name": "물놀이",
          "time": null,
          "infomation": "(4세)"
        },
        {
          "name": "열린어린이집 바다반 활동보조",
          "time": null,
          "infomation": ""
        },
        {
          "name": "빨간망토 인형극 공연 관람",
          "time": "[비대면]",
          "infomation": "(3세~5세)"
        }
      ]
    },
    {
      "date": "07",
      "activities": [
        {
          "name": "소방대피훈련",
          "time": null,
          "infomation": ""
        }
      ]
    },
    {
      "date": "08",
      "activities": [
        {
          "name": "지역사회연계활동 - 마트",
          "time": null,
          "infomation": "(4세, 5세)"
        },
        {
          "name": "열린어린이집 바다반 활동보조",
          "time": null,
          "infomation": ""
        }
      ]
    },
    {
      "date": "13",
      "activities": [
        {
          "name": "물놀이",
          "time": null,
          "infomation": "(1세)"
        },
        {
          "name": "열린어린이집 달님반 활동보조",
          "time": null,
          "infomation": ""
        }
      ]
    },
    {
      "date": "15",
      "activities": [
        {
          "name": "광복절",
          "time": null,
          "infomation": "[휴원]"
        }
      ]
    },
    {
      "date": "16",
      "activities": [
        {
          "name": "지역사회연계활동 - 마트",
          "time": null,
          "infomation": "(3세)"
        },
        {
          "name": "열린어린이집 무지개반 활동보조",
          "time": null,
          "infomation": ""
        }
      ]
    },
    {
      "date": "21",
      "activities": [
        {
          "name": "탈인형극",
          "time": null,
          "infomation": "- 바보온달과 평강공주 [주최: 아토피 천식안심학교] (0세~5세)"
        }
      ]
    },
    {
      "date": "22",
      "activities": [
        {
          "name": "비상대응훈련",
          "time": null,
          "infomation": "(태풍)"
        }
      ]
    },
    {
      "date": "23",
      "activities": [
        {
          "name": "하늘반 여름캠프",
          "time": null,
          "infomation": "(5세)"
        }
      ]
    },
    {
      "date": "27",
      "activities": [
        {
          "name": "인형극 관람",
          "time": null,
          "infomation": "'안전극 : 랑이 담배 피해야 돼요 [약물오남용/중독]' (1세~5세)"
        }
      ]
    },
    {
      "date": "30",
      "activities": [
        {
          "name": "어린이 방문교육",
          "time": null,
          "infomation": "[주최: 중랑구 어린이 사회복지급식관리지원센터] (0세~2세) - 식사예절 및 올바른 식습관 '설탕의 달콤한 보다 건강한 단 맛이 좋아요'"
        }
      ]
    }
  ],
  "etc": "생일을 축하합니다. 6일(화) 햇님반 김주원, 내민율, 이시원; 13일(화) 바다반 내소율. 국공립 천사어린이집.",
  "user_id": "string",
  "baby_id": "string"
}

```

# babydiary
## request
```bash
# request

어머니~ 우현이 멋진 우비를 입고 짠~^^하고 등장했네요 ㅎㅎ 오늘도 역시나 자동차를 가지고 왔네요 빨간차가지고 다투니까 아예 빨간 차로만 가져왔네요ㅎ 친구들 골고루나눠주고 잘 놀았어요 시소 끼적이기, 도장찍기도 하면서 즐겁게 잘지냈습니다~^^ 오전간식 요플레, 점심도 김가루하고 야무지게 먹고 양치하고 잠자리에 들었어요 기침을 간혹 하네요 집에서도 잘 관찰해 주세요.
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
  "user_id": "string",
  "baby_id": "string",
  "role": "child"
}
```

# daysummary
## request
```bash
{
  "text": "내 오늘 하루는 어땠지?",
  "session_id": "your-id"
}
```
## response
```bash
{
  "response": "답변 - ",
  "session_id": "your-id"
}
```