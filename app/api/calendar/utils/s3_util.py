# s3 연동
import boto3
import tempfile
from botocore.exceptions import ClientError
import uuid
from urllib.parse import urlparse, unquote
import os


# s3 클라이언트 설정
def set_s3_client():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    return s3_client


# s3 파일 이름 파싱
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
