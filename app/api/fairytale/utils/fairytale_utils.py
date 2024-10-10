import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from app.api.fairytale.models import StoryResponse
import os

# 알림장 요약 데이터 Load, key 선택


def select_keys_from_diary_data(data: dict):
    """
    주어진 경로에서 일기 데이터를 로드하고 선택된 키의 데이터만 반환합니다.

    Args:
        data (dict): 알림장 요약 api 반환 결과(name, emotion, activities, special 등 키워드를 포함한 딕셔너리 파일)

    Returns:
        dict: 선택된 키에 해당하는 데이터
    """

    # selected_keys = ["name", "emotion", "activities", "special"]
    selected_keys = ["name", "activities", "special", "age", "gender"]
    selected_data = {}

    for key, value in data.items():
        if key in selected_keys:
            selected_data[key] = value

    return selected_data


def load_prompts(prompt_path: str):
    """
    주어진 경로에서 프롬프트를 로드합니다.

    Args:
        prompt_path (str): 프롬프트 파일의 경로

    Returns:
        str: 로드된 프롬프트 내용
    """
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()
    return prompt


def create_pairy_chain(template: str, selected_data: dict):
    """
    주어진 템플릿과 선택된 데이터를 사용하여 Pairy 체인을 생성합니다.

    Args:
        template (str): 프롬프트 템플릿
        selected_data (dict): 선택된 데이터

    Returns:
        Chain: 생성된 Pairy 체인
    """
    selected_keys = list(selected_data.keys())
    # 모델 생성
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 출력 파서 생성
    output_parser = JsonOutputParser(pydantic_object=StoryResponse)
    format_instructions = output_parser.get_format_instructions()

    # 프롬프트 템플릿 생성
    prompt = PromptTemplate(
        input_variables=selected_keys,
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
        template=template,
    )

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


def view_generated_story(result: dict):
    """
    생성된 이야기를 콘솔에 출력합니다.

    Args:
        result (dict): 생성된 이야기 데이터
    """
    print(f"Title: {result['title']}\n")
    for i in result["pages"]:
        print(
            f"Text: \n{i['text']}\n\nIllustration Prompt: \n{i['illustration_prompt']}\n"
        )
        print("-" * 100)


def save_result_to_json(result, filename):
    """
    생성된 결과를 JSON 파일로 저장합니다.

    Args:
        result: 저장할 결과 데이터
        filename (str): 저장할 파일 이름
    """
    # 결과 디렉토리가 없으면 생성
    os.makedirs("results", exist_ok=True)

    # 결과를 JSON 파일로 저장
    with open(f"results/{filename}", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
