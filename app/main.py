from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.babydiary import babydiary
from app.api.calendar import calendar
from app.api.daysummary import daysumm
from app.api.fairytale import fairytale
from app.api.embedding import embedd
from app.api.audiomemo import audiomemo
import os


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(babydiary.router)
app.include_router(calendar.router)
app.include_router(daysumm.router)
app.include_router(fairytale.router)
app.include_router(embedd.router)
app.include_router(audiomemo.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
