# Final-ml

=======
## 캘린더 기능 사용법

### 환경설정 순서
```bash
conda create -n [ENVNAME] python=3.11.0
conda activate [ENVNAME]
pip install -r requirements.txt
```


#### If Windows:
```bash
pip install uvicorn[standard]
```
```bash
conda install -c conda-forge tesseract
```

---
### 실행 예시
- python calender.py 실행
- http://127.0.0.1:8000/docs 에서 image_path 입력 후 실행하거나 아래 명령어 새로운 터미널에서 실행.
    - result 디렉토리 없으면 생성. 모든 과정이 끝나면 해당 디렉토리안에 결과물 Json 파일 저장.
```bash
curl -X POST "http://localhost:8000/process_image" \
        -H "Content-Type: application/json" \
        -d '{"image_path": "./images/sch4.jpg"}'
```
>>>>>>> a74e5f3f09d1de7368a39cfdafbab84e6a0ba16d
