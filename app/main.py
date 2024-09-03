from fastapi import FastAPI
from app.api.babydiary import babydiary
from app.api.calendar import calendar
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.include_router(babydiary.router)
app.include_router(calendar.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
