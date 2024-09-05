import json
from uuid import uuid4
from langchain.schema import Document


def convert_data_structure(input_data):
    # Parse the input JSON string if it's a string
    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    # Create the new structure
    new_structure = {"metadata": {}, "diary": ""}

    # List of keys to be included in metadata
    metadata_keys = [
        "name",
        "emotion",
        "health",
        "nutrition",
        "activities",
        "social",
        "special",
        "keywords",
    ]

    # Copy relevant fields to metadata
    for key in metadata_keys:
        if key in input_data:
            new_structure["metadata"][key] = input_data[key]

    # Create diary content by concatenating all text fields except 'diary' and 'keywords'
    diary_content = []
    for key, value in input_data.items():
        if key not in ["diary", "keywords"] and isinstance(value, str):
            diary_content.append(f"{key}: {value}")
        elif key == "activities" and isinstance(value, list):
            diary_content.append(f"activities: {', '.join(value)}")

    new_structure["diary"] = "\n ".join(diary_content)

    return new_structure


def preprocess_metadata(metadata):
    """
    리스트를 쉼표로 구분된 문자열로 변환하고, 다른 복잡한 데이터 타입을 문자열로 변환합니다.
    한글은 그대로 유지합니다.
    """
    for key, value in metadata.items():
        if isinstance(value, list):
            metadata[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            metadata[key] = str(value)  # 딕셔너리의 경우 단순 문자열로 변환
    return metadata


def create_document_from_data(data):
    """
    딕셔너리 형태의 데이터로부터 Document 객체를 생성합니다.
    """
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary")

    metadata = data["metadata"]
    metadata["source"] = "diary"
    metadata = preprocess_metadata(metadata)  # 메타데이터 전처리
    page_content = data["diary"]
    return Document(page_content=page_content, metadata=metadata)
