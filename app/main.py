from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Annotated
from pydantic import BaseModel, HttpUrl         #HttpUrl - adicional
from joblib import load
import json
from Server_PD import download_model_minio
# Adicionales
from get_video_info_api import get_info
import re
from train import model_kids, model_genres

from pydantic import BaseModel
# Ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    app.state.model_genre = (download_model_minio("pd1", "grupo1/models/genres/genres_definitive", claves))
    app.state.encoder_genre = (download_model_minio("pd1", "grupo1/models/genres/encoder", claves)) # encoder
    app.state.model_kids = (download_model_minio("pd1", "grupo1/models/kids/kids_definitive", claves))
    app.state.pipe_kids = (download_model_minio("pd1", "grupo1/models/kids/pipe_kids", claves))
    app.state.pipe_genres = (download_model_minio("pd1", "grupo1/models/genres/pipe_genres", claves))
    yield

#Hay que definir el BaseModel
app = FastAPI(lifespan=lifespan)

#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

class VideoInput(BaseModel):
    url: HttpUrl

class GenrePrediction(BaseModel):
    genre: str
    #confidence: float

class KidsPrediction(BaseModel):
    safe: bool

@app.get("/video/genre", response_class=HTMLResponse)
def genre_page(request: Request):
    return templates.TemplateResponse(request, name="genres.html")

@app.get("/video", response_class=HTMLResponse)
def video_page(request: Request):
    return templates.TemplateResponse(request, name="video_check.html")

# Predicciones
@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, name='index.html')


@app.post("/video/genre", response_model=GenrePrediction)
async def predict_genre(video: VideoInput):
    """Calsifica segun el género del video"""
    modelo = model_genres(app.state.model_genre, app.state.pipe_genres, app.state.encoder_genre)
    genre_name = modelo._get_data_and_predict(str(video.url))

    if genre_name is None:
        return GenrePrediction(genre="Error: URL no válida", confidence=0.0)
        
    return GenrePrediction(genre=str(genre_name))

@app.post("/video/check", response_model=KidsPrediction)
async def predict_kids(video: VideoInput):
    """Clasifica si es apto para niños"""
    modelo = model_kids(app.state.model_kids, app.state.pipe_kids)
    label = modelo._get_data_and_predict(str(video.url))

    if label is None:
        return KidsPrediction(safe=False) 
    
    is_safe = bool(label)
    return KidsPrediction(safe=is_safe)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=2350, reload=True)
