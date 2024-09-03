from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.langsmith("babydiary")

# Initialize FastAPI app
app = FastAPI()


# Pydantic model for the daycare report
class DaycareReport(BaseModel):
    name: str = Field(description="Child's name")
    emotion: str = Field(description="Child's overall mood and emotional state")
    health: str = Field(description="Child's physical health and well-being")
    nutrition: str = Field(description="Summary of child's meals and eating behavior")
    activities: List[str] = Field(description="Main activities the child engaged in")
    social: str = Field(description="Child's interactions with peers and teachers")
    special: str = Field(description="Special achievements or unusual occurrences")
    keywords: List[str] = Field(
        description="Important keywords from entities other than name entities"
    )


# Set up the parser and prompts
output_parser = JsonOutputParser(pydantic_object=DaycareReport)

# You need to define these in your code or import them
from prompts.prompts import template, generate_diary

prompt = PromptTemplate(
    input_variables=["report"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    template=template,
)

# Initialize the model and chains
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
chain = prompt | model | output_parser
chain2 = (lambda x: generate_diary(x)) | model | StrOutputParser()


# FastAPI route for processing daycare reports
@app.post("/generate_diary")
async def process_report(input_notice: str):
    try:
        report = chain.invoke({"report": input_notice})
        result = chain2.invoke(report)
        report["diary"] = result

        # Save results to a file
        if not os.path.exists("results"):
            os.makedirs("results")
        with open(f"results/results_0902.json", "w", encoding="utf-8") as json_file:
            json.dump(report, json_file, ensure_ascii=False, indent=4)

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
