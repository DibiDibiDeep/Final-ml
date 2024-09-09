from openai import OpenAI


def generate_image(client, context):
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
