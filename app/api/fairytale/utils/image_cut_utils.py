import cv2
import numpy as np
import requests
import os

def detect_and_crop_panels(image_url):
    # 이미지 URL에서 다운로드
    response = requests.get(image_url)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # image = cv2.imread(image_url)
    
    if image is None:
        raise ValueError(f"Unable to read image from {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 이진화
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 면적 기준으로 정렬 (큰 것부터)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    
    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        panel = image[y:y+h, x:x+w]
        panels.append(panel)
    
    # 패널을 왼쪽 상단부터 시계방향으로 정렬
    panels = sorted(panels, key=lambda p: (p[0][0][1], p[0][0][0]))
    
    return panels

#### 이미지 컷 테스트를 위한 함수.(S3에서 이미지 다운로드 후 패널로 분할)
# S3 URL 파싱
def parse_s3_url(url):
        # URL 파싱
        parsed_url = urlparse(url)

        # 경로에서 파일 이름 추출
        file_name = os.path.basename(unquote(parsed_url.path))

        # 타임스탬프와 원본 파일 이름 분리 (옵션)
        parts = file_name.split("-", 1)
        timestamp = parts[0] if len(parts) > 1 else None
        original_file_name = parts[1] if len(parts) > 1 else file_name

        return {
            "full_file_name": file_name,
            "timestamp": timestamp,
            "original_file_name": original_file_name,
        }

# s3 클라이언트 설정
def set_s3_client():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    return s3_client


if __name__ == '__main__':
    # 테스트 이미지 -> S3에서 다운로드 -> 패널로 분할 -> 결과 확인 및 저장(/home/insu/final-project/Final-ml/app/api/fairytale/utils/images)
    from dotenv import load_dotenv
    import boto3
    from urllib.parse import urlparse, unquote
    
    load_dotenv()
    s3_client = set_s3_client()
    bucket = os.getenv("AWS_S3_BUCKET")

    image_path = os.getenv("TEST_IMAGE_PATH")
    key = parse_s3_url(str(image_path))["full_file_name"]

    save_path = f"./images/{key}"

    # Check if the file already exists locally
    if not os.path.exists(save_path):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download the file from S3
        s3_client.download_file(bucket, key, save_path)
        print(f"Downloaded file from S3 to {save_path}")
    else:
        print(f"File already exists at {save_path}")
  
    panels = detect_and_crop_panels(save_path)

    # 결과 저장
    for i, panel in enumerate(panels):
        cv2.imwrite(f'./images/panel_{i+1}.png', panel)
    
    print(f"Successfully saved {len(panels)} panels.")