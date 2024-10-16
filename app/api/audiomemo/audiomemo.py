from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
import os
import io

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=apikey)


router = APIRouter()


@router.post("/whisper")
async def whisper(file: UploadFile = File(...)):
    try:
        # audio file read
        audio_data = await file.read()
        buffer = io.BytesIO(audio_data)
        buffer.name = "temp.mp3"  # openai가 파일 이름에서 형식을 추출해서 사용

        # whisper api call
        print("Get Text...")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=buffer, language="ko"
        )

        result_text = transcription.text

        print(
            f"""
Result: 
        {result_text}
                """
        )
        return JSONResponse(status_code=200, content={"transcription": result_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
