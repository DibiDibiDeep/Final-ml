import re
from typing import Dict, List, Union


class DateProcessor:
    """
    날짜 처리를 위한 클래스입니다.
    텍스트 파싱, 날짜 정규화, 이벤트 데이터 처리 등의 기능을 제공합니다.
    """

    DATE_PATTERN = r"((\d{4})년\s*)?(\d{1,2})월\s*(\d{1,2})일\s*(\([^\)]+\))?\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)?"

    def __init__(self, data: Union[str, Dict[str, List[Dict[str, str]]]]):
        """
        DateProcessor 클래스의 생성자입니다.

        Args:
            data (Union[str, Dict[str, List[Dict[str, str]]]]): 처리할 데이터. 문자열 또는 이벤트 데이터 딕셔너리.
        """
        self.data = data
        self.processed_data = None

    def process(self) -> Union[str, Dict[str, List[Dict[str, str]]]]:
        """
        입력된 데이터를 처리합니다.

        Returns:
            Union[str, Dict[str, List[Dict[str, str]]]]: 처리된 데이터
        """
        if isinstance(self.data, str):
            self.processed_data = self.parse_text_by_date(self.data)
        elif isinstance(self.data, dict):
            self.processed_data = self.normalize_event_dates(self.data)
        return self.processed_data

    @classmethod
    def parse_text_by_date(cls, text: str) -> str:
        """
        텍스트를 날짜형식을 기준으로 파싱합니다.

        Args:
            text (str): 파싱할 텍스트

        Returns:
            str: 날짜형식을 기준으로 파싱된 텍스트
        """
        matches = list(re.finditer(cls.DATE_PATTERN, text))
        parsed_sections = cls._parse_sections(text, matches)
        return "\n".join(parsed_sections)

    @classmethod
    def _parse_sections(cls, text: str, matches: List[re.Match]) -> List[str]:
        """
        텍스트를 날짜 패턴에 따라 섹션으로 나눕니다.

        Args:
            text (str): 원본 텍스트
            matches (List[re.Match]): 날짜 패턴 매치 리스트

        Returns:
            List[str]: 파싱된 섹션 리스트
        """
        sections = []
        last_end = 0

        for match in matches:
            if match.start() > last_end:
                sections.append(text[last_end : match.start()].strip())

            date = cls._format_date(match)
            content_end = cls._find_content_end(text, match)
            content = text[match.end() : content_end].strip()
            sections.append(f"{date}: {content}")

            last_end = content_end

        if last_end < len(text):
            sections.append(text[last_end:].strip())

        return sections

    @classmethod
    def _format_date(cls, match: re.Match) -> str:
        """
        매치된 날짜를 포맷팅합니다.

        Args:
            match (re.Match): 날짜 패턴 매치 객체

        Returns:
            str: 포맷팅된 날짜 문자열
        """
        day = cls.normalize_date_string(match.group(4))
        return day

    @classmethod
    def _find_content_end(cls, text: str, match: re.Match) -> int:
        """
        현재 날짜에 해당하는 내용의 끝 위치를 찾습니다.

        Args:
            text (str): 원본 텍스트
            match (re.Match): 현재 날짜 패턴 매치 객체

        Returns:
            int: 내용의 끝 위치
        """
        next_match = re.search(cls.DATE_PATTERN, text[match.end() :])
        return match.end() + next_match.start() if next_match else len(text)

    @staticmethod
    def normalize_date_string(date_str: str) -> str:
        """
        날짜 문자열을 정규화합니다.

        Args:
            date_str (str): 정규화할 날짜 문자열

        Returns:
            str: 정규화된 날짜 문자열
        """
        if date_str.isdigit():
            return date_str.zfill(2)  # 이미 숫자인 경우 그대로 반환 (2자리로 패딩)
        date_str = date_str.replace("일", "").strip()
        return date_str.zfill(2)  # 한 자리 숫자를 두 자리로 변환 (예: '5' -> '05')

    @classmethod
    def normalize_event_dates(
        cls, data: Dict[str, List[Dict[str, str]]]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        이벤트 데이터의 날짜를 정규화합니다.

        Args:
            data (Dict[str, List[Dict[str, str]]]): 정규화할 이벤트 데이터

        Returns:
            Dict[str, List[Dict[str, str]]]: 날짜가 정규화된 이벤트 데이터
        """
        for event in data["events"]:
            if isinstance(event["date"], int):
                event["date"] = str(event["date"]).zfill(
                    2
                )  # 정수형 날짜는 문자열로 변환하고 2자리로 패딩
            else:
                match = re.search(cls.DATE_PATTERN, event["date"])
                if match:
                    event["date"] = cls.normalize_date_string(match.group(4))
                else:
                    event["date"] = cls.normalize_date_string(event["date"])
        return data


# 사용 예시
if __name__ == "__main__":
    # 텍스트 처리 예시
    text_data = """
    8월 15일 광복절
    8월 22일 목요일 회의
    23일 (금) 팀 빌딩 활동
    2024년 8월 31일 토 월간 보고서 제출
    9월 1일 일요일 새 프로젝트 시작
    """
    text_processor = DateProcessor(text_data)
    processed_text = text_processor.process()
    print("처리된 텍스트:")
    print(processed_text)

    # 이벤트 데이터 처리 예시
    event_data = {
        "events": [
            {"date": "8월 15일", "description": "광복절"},
            {"date": "22일 목요일", "description": "회의"},
            {"date": "2024년 8월 31일 토요일", "description": "보고서 제출"},
            {"date": "1일", "description": "새 프로젝트 시작"},
            {"date": 24, "description": "정수형 날짜 테스트"},
        ]
    }
    event_processor = DateProcessor(event_data)
    processed_events = event_processor.process()
    print("\n처리된 이벤트 데이터:")
    for event in processed_events["events"]:
        print(f"{event['date']}: {event['description']}")
