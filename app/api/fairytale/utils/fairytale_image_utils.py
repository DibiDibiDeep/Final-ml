from typing import List
from openai import OpenAI
import pyshorteners
from app.api.fairytale.models import StoryResponse, StoryPage


def generate_images(client: OpenAI, story_response: StoryResponse) -> List[str]:
    # 책 표지 이미지 생성
    cover_image_url = generate_image(client, story_response.book_cover_description)

    # 각 페이지의 이미지 생성
    page_image_urls = [
        generate_image(client, page.illustration_prompt)
        for page in story_response.pages
    ]

    # 모든 이미지 URL을 리스트로 반환 (표지 이미지가 첫 번째)
    return [cover_image_url] + page_image_urls


def generate_image(client: OpenAI, context: str) -> str:
    # DALL-E 3 모델을 사용하여 이미지 생성
    response = client.images.generate(
        model="dall-e-3",
        prompt=context,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # 생성된 이미지의 URL을 저장
    image_url = response.data[0].url
    return image_url


def shorten_url(long_url):
    try:
        s = pyshorteners.Shortener()
        short_url = s.tinyurl.short(long_url)
        return short_url
    except Exception as e:
        print(f"URL 단축 실패: {str(e)}")
        return long_url  # 단축 실패 시 원래 URL 반환
