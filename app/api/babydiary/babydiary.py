import json, os
from fastapi import APIRouter, HTTPException
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from app.api.babydiary.models import DaycareReport
from app.api.babydiary.prompts import template, generate_diary


# Set up the parser and prompts
output_parser = JsonOutputParser(pydantic_object=DaycareReport)

# You need to define these in your code or import them

prompt = PromptTemplate(
    input_variables=["report"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template=template,
)

# Initialize the model and chains
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
chain = prompt | model | output_parser
chain2 = (lambda x: generate_diary(x)) | model | StrOutputParser()

router = APIRouter()


@router.post("/generate_diary")
async def process_report(input_notice: str):
    from langchain_teddynote import logging

    logging.langsmith("babydiary")

    try:
        report = chain.invoke({"report": input_notice})
        result = chain2.invoke(report)
        report["diary"] = result

        # 결과를 json 파일로 저장(or DB 저장(추후))
        if not os.path.exists("reseults/"):
            os.makedirs("results/")
        with open(f"results/diary_result", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
