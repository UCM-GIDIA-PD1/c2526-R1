from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
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
    return templates.TemplateResponse("video_check.html", {"request": request})

class GenreRequest(BaseModel):
    url: str

class GenreResponse(BaseModel):
    genre: str
    confidence: float

@app.post("/video/genre", response_model=GenreResponse)
def classify_video(data: GenreRequest):
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
    return templates.TemplateResponse("video_genre.html", {"request": request})