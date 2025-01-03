from threading import Thread
import json
from queue import Queue
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import logging

from .parsers import extract_json, extract_list, rectangle_corners
from .wrappers import (
    job_easy_ocr,
    job_easy_ocr_boxes,
    job_tesseract,
    job_tesseract_boxes,
)


# def wrapper(func, args, queue):
#     queue.put(func(args))
class OCRJobFailedError(Exception):
    pass


def wrapper(func, args, queue):
    try:
        result = func(args)
        if result is None or result == "":
            raise OCRJobFailedError(f"OCR job {func.__name__} failed")
        queue.put(result)
    except Exception as e:
        queue.put(e)


# custom error
class NoTextDetectedError(Exception):
    pass


def detect():
    """Unimplemented"""
    raise NotImplementedError


def detect_async():
    """Unimplemented"""
    raise NotImplementedError


def get_jobs(languages: list[str], boxes=False):
    jobs = [
        job_easy_ocr if not boxes else job_easy_ocr_boxes,
        job_tesseract if not boxes else job_tesseract_boxes,
    ]
    # ko or en in languages
    if "ko" in languages or "en" in languages:
        try:
            if not boxes:
                from .wrappers.easy_pororo_ocr import job_easy_pororo_ocr

                jobs.append(job_easy_pororo_ocr)
            else:
                from .wrappers.easy_pororo_ocr import job_easy_pororo_ocr_boxes

                jobs.append(job_easy_pororo_ocr_boxes)
        except ImportError as e:
            print(e)
            print(
                "[!] Pororo dependencies is not installed. Skipping Pororo (EasyPororoOCR)."
            )
            pass
    return jobs


def detect_text(
    image_path: str,
    lang: list[str],
    context: str = "",
    tesseract: dict = {},
    openai: dict = {"model": "gpt-4"},
):
    """Detect text from an image using EasyOCR and Tesseract, then combine and correct the results using OpenAI's LLM."""
    options = {
        "path": image_path,  # "demo.png",
        "lang": lang,  # ["ko", "en"]
        "context": context,
        "tesseract": tesseract,
        "openai": openai,
    }
    jobs = get_jobs(languages=options["lang"], boxes=False)

    queues = []
    for job in jobs:
        queue = Queue()
        Thread(target=wrapper, args=(job, options, queue)).start()
        queues.append(queue)

    # results = [queue.get() for queue in queues]
    results = []
    for queue in queues:
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        results.append(result)

    result_indexes_prompt = ""  # "[0][1][2]"
    result_prompt = ""  # "[0]: result_0\n[1]: result_1\n[2]: result_2"

    for i in range(len(results)):
        result_indexes_prompt += f"[{i}]"
        result_prompt += f"[{i}]: {results[i]}"

        if i != len(results) - 1:
            result_prompt += "\n"

    optional_context_prompt = (
        f"[context]: {options['context']}" if options["context"] else ""
    )

    # prompt = f"""Combine and correct OCR results {result_indexes_prompt}, using \\n for line breaks. Langauge is in {'+'.join(options['lang'])}. Remove unintended noise. Refer to the [context] keywords. Answer in the JSON format {{data:<output:string>}}:
    # {result_prompt}
    # {optional_context_prompt}"""

    prompt = f"""First, determine if this document is related to a daycare center schedule:
    1. If it's not related to daycare schedules (lacks specific daycare schedule information or is clearly about a different topic), output "no" in JSON format {{data:"no"}}.
    2. If it is related to daycare schedules, combine and correct OCR results {result_indexes_prompt}, using \\n for line breaks. Language is in {'+'.join(options['lang'])}. Remove unintended noise. Refer to the [context] keywords. Answer in the JSON format {{data:<output:string>}}:
    {result_prompt}
    {optional_context_prompt}"""

    prompt = prompt.strip()

    # Prioritize user-specified API_KEY
    api_key = options["openai"].get("API_KEY", os.environ.get("OPENAI_API_KEY"))

    # Make a shallow copy of the openai options and remove the API_KEY
    openai_options = options["openai"].copy()
    if "API_KEY" in openai_options:
        del openai_options["API_KEY"]

    chat_model = ChatOpenAI(
        model_name=openai_options.get("model", "gpt-4"),
        temperature=openai_options.get("temperature", 0),
        api_key=api_key,
    )

    message = HumanMessage(content=prompt)
    response = chat_model([message])
    output = response.content

    logging.info(f" BetterOCR LLM completed")

    result = extract_json(output)
    print(result)

    if "data" in result:
        if result["data"].strip().lower() == "no":
            raise ValueError("Not related to daycare schedules")
        return result["data"]
    if isinstance(result, str):
        return result
    raise NoTextDetectedError("No text detected")


def detect_text_async():
    """Unimplemented"""
    raise NotImplementedError


def detect_boxes(
    image_path: str,
    lang: list[str],
    context: str = "",
    tesseract: dict = {},
    openai: dict = {"model": "gpt-4"},
):
    options = {
        "path": image_path,  # "demo.png",
        "lang": lang,  # ["ko", "en"]
        "context": context,
        "tesseract": tesseract,
        "openai": openai,
    }
    jobs = get_jobs(languages=options["lang"], boxes=True)

    queues = []
    for job in jobs:
        queue = Queue()
        Thread(target=wrapper, args=(job, options, queue)).start()
        queues.append(queue)

    results = [queue.get() for queue in queues]
    logging.info(f"[*] OCR completed")
    result_indexes_prompt = ""  # "[0][1][2]"
    result_prompt = ""  # "[0]: result_0\n[1]: result_1\n[2]: result_2"

    for i in range(len(results)):
        result_indexes_prompt += f"[{i}]"

        boxes = results[i]
        boxes_json = json.dumps(boxes, ensure_ascii=False, default=int)

        result_prompt += f"[{i}]: {boxes_json}"

        if i != len(results) - 1:
            result_prompt += "\n"

    optional_context_prompt = (
        " " + "Please refer to the keywords and spelling in [context]"
        if options["context"]
        else ""
    )
    optional_context_prompt_data = (
        f"[context]: {options['context']}" if options["context"] else ""
    )

    template = """Combine and correct OCR data {result_indexes}. 
    Include many items as possible. Langauge is in {languages}(Avoid arbitrary translations). 
    Remove unintended noise.{optional_context_prompt} Answer in the JSON format. 
    Ensure coordinates are integers (round based on confidence if necessary) and output in the same JSON format (indent=0): 
        Array({{box:[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],text:str}}):
            {ocr_results}
    {optional_context_prompt_data}"""

    template = template.strip()

    # Prioritize user-specified API_KEY
    api_key = options["openai"].get("API_KEY", os.environ.get("OPENAI_API_KEY"))

    # Make a shallow copy of the openai options and remove the API_KEY
    openai_options = options["openai"].copy()
    if "API_KEY" in openai_options:
        del openai_options["API_KEY"]

    output_parser = JsonOutputParser()
    prompt = PromptTemplate(
        input_variables=[
            "result_indexes",
            "languages",
            "optional_context_prompt",
            "ocr_results",
            "optional_context_prompt_data",
        ],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
        template=template,
    )
    client = ChatOpenAI(api_key=api_key, **openai_options)
    chain = prompt | client | output_parser

    logging.info(f"[*] LLM Start")
    result = chain.invoke(
        {
            "result_indexes": result_indexes_prompt,
            "languages": "+".join(options["lang"]),
            "ocr_results": result_prompt,
            "optional_context_prompt": optional_context_prompt,
            "optional_context_prompt_data": optional_context_prompt_data,
        }
    )

    logging.info(f"[*] LLM End")
    return result


def detect_boxes_async():
    """Unimplemented"""
    raise NotImplementedError
