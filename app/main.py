# Bibliotecas
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Annotated
from pydantic import BaseModel, HttpUrl         #HttpUrl - adicional
from joblib import load
# Adicionales
from extraccion.get_video_info_api import get_info
import re
import os

# Ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_genre = load('pd1/grupo1/models/genres/genres_definitive.joblib')
    app.state.encoder_genre = load('pd1/grupo1/models/genres/encoder.joblib') # encoder
    app.state.model_kids = load('pd1/grupo1/models/kids/kids_definitive.joblib')
    yield

# Web
app = FastAPI(lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

 #Entrada y salida de datos
class VideoInput(BaseModel):
    url: HttpUrl

class GenrePrediction(BaseModel):
    genre: str
    confidence: float

class KidsPrediction(BaseModel):
    safe: bool


# Extraccion de datos 
def _get_data_and_predict(model, url: str):
    """
    Parametros entrada -> 
    model: Modelo correspondiente a la predicción que queramos hacer. (Generos o kids).
    url (string): Url del video que se quiere clasficar.

    Parametros salida -> 
        Predicción (string)

    Extrae los datos de un video a partir de la url y predice el resultado
    """
    # Extracción de ID
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not video_id_match:
        return None, 0.0
    
    video_id = video_id_match.group(1)

    # Extraemos los datos del video a partir del ID
    df_video = get_info(video_id)
    
    if df_video is None or df_video.empty:
        return None, 0.0

    # Predicción
    prediction= model.predict(df_video)[0]

    try:
        prob = model.predict_proba(df_video).max()
    except:
        prob = 0.95
    
    return prediction, prob


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
    label, prob = _get_data_and_predict(app.state.model_genre, str(video.url))

    if label is None:
        return GenrePrediction(genre="Error: URL no válida", confidence=0.0)

    genre_name = app.state.encoder_genre.inverse_transform([label])[0]
    
    return GenrePrediction(genre=str(genre_name), confidence=float(prob))

@app.post("/video/check", response_model=KidsPrediction)
async def predict_kids(video: VideoInput):
    """Clasifica si es apto para niños"""
    label, prob = _get_data_and_predict(app.state.model_kids, str(video.url))

    if label is None:
        return KidsPrediction(safe=False) 
    
    is_safe = bool(label)
    return KidsPrediction(safe=is_safe)

