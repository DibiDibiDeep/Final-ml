from fastapi import APIRouter, HTTPException
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from models.babydiary import DaycareReport


# Set up the parser and prompts
output_parser = JsonOutputParser(pydantic_object=DaycareReport)

# You need to define these in your code or import them
from .prompts import template, generate_diary

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

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
