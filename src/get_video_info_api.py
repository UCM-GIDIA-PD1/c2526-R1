# Para la extracción de videos con la API
from googleapiclient.discovery import build
from yt_dlp import YoutubeDL
import pprint as pprint
import os
import json
import re
import pandas as pd

from urllib import response

def ini_youtube():
    with open("src/Private/Claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
        API_KEY = claves["Clave_API"]
    
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    return youtube

def clean_vtt(text):
    text = re.sub(r"WEBVTT.*\n", "", text)
    text = re.sub(r"\d+:\d+:\d+\.\d+ --> .*", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def clean_vtt_smart(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not lines or not line in lines[-1]:
            lines.append(line)
    return " ".join(lines)

def get_info(id_video):
    """ Saca la información del video, si hay subtítulos los limpia, y añade dichos datos al dataframe. """
    youtube = ini_youtube()
    # Inicializamos la variable de los subtitulos
    cleant_sub = None

    # Hacemos la llamada a la API para obtener los detalles del video
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics,status,topicDetails,recordingDetails",
        id=id_video
    )  

    # Ejecutamos la solicitud
    response = request.execute()

    if not response["items"]:
        return None
    
    video = response["items"][0]

    has_captions = video["contentDetails"].get("caption") in ["true", True]

    # --- SUBTÍTULOS ---
    if (video["contentDetails"].get("caption") == "true"):
        url =  "https://www.youtube.com/watch?v=" + id_video
        # Opciones de descarga
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,      # subtítulos automáticos
            "subtitleslangs": ["en"], # idioma
            "subtitlesformat": "vtt",       # formato
            "outtmpl": f"subs/{id_video}.%(ext)s",
            "quiet": True,
            "no_warnings": True
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        vtt_file = f"subs/{id_video}.en.vtt"

        if os.path.exists(vtt_file):
            with open(vtt_file, "r", encoding="utf-8") as f:
                subtitles = f.read()

            clean_sub = clean_vtt(subtitles)
            cleant_sub = clean_vtt_smart(clean_sub)

            # Borramos el archivo de subtítulos descargado tras limpiarlo
            # os.remove(vtt_file)
    # --- GENERO  ---
    youtube_categories = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers"
}
    # --- GENEROS ---
    generos = video.get("topicDetails", {}).get("topicCategories", [])

    if generos:
        generos = [
            genre.split("/")[-1].replace("_", " ")
            for genre in generos
        ]
        generos_str = ", ".join(generos)
    
    else:
        generos_str = "None"

    # --- DATAFRAME ---
    df_video = pd.DataFrame({
        "ID": id_video,
        "Titulo": video["snippet"]["title"],
        "Descripcion": video["snippet"]["description"],
        "Visualizaciones": video["statistics"]["viewCount"],
        "Tags":", ".join(video["snippet"].get("tags", [])),
        "Duracion": video["contentDetails"]["duration"],
        "Fecha_publicacion": video["snippet"]["publishedAt"],
        "Titulo_canal": video["snippet"]["channelTitle"],
        "Subtitulos": cleant_sub if has_captions and cleant_sub else "None",
        "Genero": youtube_categories[video["snippet"]["categoryId"]], 
        "Generos": generos_str,
        #"ContentRating": content_rating_str,
        "Made for kids": video["status"]["madeForKids"],
        "Rango_edad": "Adult"
    }, index=[0])

    return df_video
