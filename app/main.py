from fastapi import FastAPI

from app.api.babydiary import babydiary
from app.api.calendar import calendar
from app.api.daysummary import daysumm



app = FastAPI()


app.include_router(babydiary.router)
app.include_router(calendar.router)
app.include_router(daysumm.router)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)  #  host="127.0.0.1"

