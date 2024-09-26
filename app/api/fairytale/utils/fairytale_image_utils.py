import pyshorteners
import logging
import asyncio
import time
from requests.exceptions import RequestException
import os
from openai import AsyncOpenAI

image_model = os.getenv("IMAGE_MODEL")

# 이미지 생성 함수
async def generate_image(client: AsyncOpenAI, prompt: str):

    # DALL-E 3 모델을 사용하여 이미지 생성
    response = await client.images.generate(
        model= image_model,
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

    logging.info("Title image generation started")

    # 표지 이미지 생성 프롬프트 생성(캐릭터 정보 + 표지 이미지 생성 프롬프트)
    title_prompt = f"{characters_info} {book_cover_description}"
    title_img_path = await generate_image(client, title_prompt)
    # url 짧게 변경
    title_short_url = await shorten_url(title_img_path)

    logging.info("Title image generation completed")

    return title_short_url

# 페이지별 이미지 생성을 비동기로 처리
async def generate_page_image(client: AsyncOpenAI, characters_info: str, page: dict, index: int):

    logging.info(f"Page image generation started for page {index}")

    # 페이지 이미지 생성 프롬프트 생성(캐릭터 정보 + 페이지 이미지 생성 프롬프트)
    page_prompt = f"{characters_info}{page['illustration_prompt']}"
    image_url = await generate_image(client, page_prompt)
    # url 짧게 변경
    short_url = await shorten_url(image_url)

    logging.info(f"Page image generation completed for page {index}")

    return index, {"image_url": short_url, **page}

# 모든 이미지(표지 + 페이지) 생성 함수
async def generate_all_images(client: AsyncOpenAI, characters_info: str, book_cover_description: str, pages: list):
    # 이미지 생성 시작 시간 기록
    start_time = time.time()
    
    # 표지 이미지 생성
    cover_task = generate_cover_image(client, characters_info, book_cover_description)
    
    # 페이지 이미지 생성
    page_tasks = [generate_page_image(client, characters_info, page, i) for i, page in enumerate(pages)]
    
    # 모든 태스크 실행
    all_results = await asyncio.gather(cover_task, *page_tasks)
    
    # 최소 1분 보장(Too Many Requests Error 방지)
    elapsed_time = time.time() - start_time
    # if elapsed_time < 60:
    #     await asyncio.sleep(60 - elapsed_time)
    MAX_EXECUTION_TIME = 60
    MIN_EXECUTION_TIME = 55  # 최소 실행 시간을 설정

    if elapsed_time < MIN_EXECUTION_TIME:
        await asyncio.sleep(min(MIN_EXECUTION_TIME - elapsed_time, MAX_EXECUTION_TIME - elapsed_time))
    # 결과 정리
    cover_image = all_results[0]
    page_images = all_results[1:]
    logging.info("All images generation completed")
    return cover_image, page_images


# 이미지 URL 단축 함수
async def shorten_url(url, max_retries=3, timeout=5):
    s = pyshorteners.Shortener(timeout=timeout)
    
    for attempt in range(max_retries):
        try:
            return s.tinyurl.short(url)
        except RequestException as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to shorten URL after {max_retries} attempts: {e}")
                return url  # 실패 시 원본 URL 반환
            await asyncio.sleep(1)  # 재시도 전 1초 대기
    logging.info(f"URL:{url}")
    return url  # 모든 시도 실패 시 원본 URL 반환

