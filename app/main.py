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
from comun.Server_PD import download_model_minio
# Adicionales
from extraccion.get_video_info_api import get_info
import re

from pydantic import BaseModel
# Ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    app.state.model_genre = load(download_model_minio("pd1", "grupo1/models/genres/genres_definitive.joblib", claves))
    app.state.encoder_genre = load(download_model_minio("pd1", "grupo1/models/genres/encoder.joblib", claves)) # encoder
    app.state.model_kids = load(download_model_minio("pd1", "grupo1/models/kids/kids_definitive.joblib", claves))
    yield

#Hay que definir el BaseModel
app = FastAPI()

templates = Jinja2Templates(directory="templates")
class VideoRequest(BaseModel):
    url: str

class VideoResponse(BaseModel):
    safe: bool

@app.post("/video/check", response_model=VideoResponse)
def check_video(data: VideoRequest):
    print(data.url)
    #Simulación para probar web
    if "kids" in data.url.lower():
        return VideoResponse(safe=True)
    else:
        return VideoResponse(safe=False)
    
@app.get("/video", response_class=HTMLResponse)
def video_page(request: Request):
    return templates.TemplateResponse(request,
                                      name = "video_check.html",
                                      context = {"request": request})

class GenreResponse(BaseModel):
    genre: str
    confidence: float

@app.post("/video/genre", response_model=GenreResponse)
def classify_video(data: VideoRequest):
    url = data.url.lower()

    #Simulación para probar web
    if "music" in url:
        return GenreResponse(genre="Música", confidence=0.92)
    elif "game" in url:
        return GenreResponse(genre="Videojuegos", confidence=0.88)
    elif "edu" in url:
        return GenreResponse(genre="Educación", confidence=0.90)
    elif "sport" in url:
        return GenreResponse(genre="Deportes", confidence=0.85)
    else:
        return GenreResponse(genre="Entretenimiento", confidence=0.75)
    
@app.get("/video/genre", response_class=HTMLResponse)
def genre_page(request: Request):
    return templates.TemplateResponse(request, 
                                      name = "genres.html", 
                                      context = {"request": request})