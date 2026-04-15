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

# Ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargamos los modelos finales
    app.state.model_genre = load('models/final_model_genre.joblib')
    app.state.model_kids = load('models/final_model_kids.joblib')
    yield

# Web
app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

 #Entrada y salida de datos
class VideoInput(BaseModel):
    url: HttpUrl

class Prediction(BaseModel):
    url: str
    prediction: str



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
        return "Error: URL no válida", 0.0
    
    video_id = video_id_match.group(1)

    # Extraemos los datos del video a partir del ID
    df_video = get_info(video_id)
    
    if df_video is None or df_video.empty:
        return "Error: No se pudo obtener info del video", 0.0

    # Predicción
    prediction_encoded = model.predict(df_video)                #[0]
    
    return str(prediction_encoded)


# Predicciones
@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, name='index.html')

@app.post("/predict/genre", response_model=Prediction)
async def predict_genre(video: VideoInput):
    """Calsifica segun el género del video"""
    label = _get_data_and_predict(app.state.model_genre, str(video.url))
    return Prediction(url=str(video.url), prediction=label)

@app.post("/predict/kids-safe", response_model=Prediction)
async def predict_kids(video: VideoInput):
    """Clasifica si es apto para niños"""
    label = _get_data_and_predict(app.state.model_kids, str(video.url))
    return Prediction(url=str(video.url), prediction=label)

