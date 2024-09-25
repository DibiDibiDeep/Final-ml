from typing import List
from openai import OpenAI
import pyshorteners
from app.api.fairytale.models import StoryResponse, StoryPage

import asyncio
from openai import AsyncOpenAI

import requests
from requests.exceptions import RequestException


async def generate_image(client: AsyncOpenAI, prompt: str):

    # DALL-E 3 모델을 사용하여 이미지 생성
    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # 생성된 이미지의 URL을 저장
    image_url = response.data[0].url
    return image_url

# 표지 이미지 생성 함수
async def generate_cover_image(client: AsyncOpenAI, characters_info: str, book_cover_description: str):
    print("표지 이미지 생성 시작")
    # title_prompt = f"{characters_info} {result['book_cover_description']}"
    title_prompt = f"{characters_info} {book_cover_description}"
    title_img_path = await generate_image(client, title_prompt)
    title_short_url = await shorten_url(title_img_path)
    print("표지 이미지 생성 완료")
    return title_short_url

# 페이지별 이미지 생성을 비동기로 처리
async def generate_page_image(client: AsyncOpenAI, characters_info: str, page: dict, index: int):
    page_prompt = f"{characters_info}{page['illustration_prompt']}"
    image_url = await generate_image(client, page_prompt)
    short_url = await shorten_url(image_url)
    return index, {"image_url": short_url, **page}
    
async def shorten_url(url, max_retries=3, timeout=5):
    s = pyshorteners.Shortener(timeout=timeout)
    
    for attempt in range(max_retries):
        try:
            return s.tinyurl.short(url)
        except RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to shorten URL after {max_retries} attempts: {e}")
                return url  # 실패 시 원본 URL 반환
            await asyncio.sleep(1)  # 재시도 전 1초 대기

    return url  # 모든 시도 실패 시 원본 URL 반환

