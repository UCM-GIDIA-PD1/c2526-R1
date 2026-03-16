import requests
import re
import json
from bs4 import BeautifulSoup
from wonderwords import RandomWord
import random
import datetime
from tqdm import tqdm
import time

from extraccion.get_video_ids_kids import get_vkids_ids

def get_video_ids(query):
    '''
    Devuelve una lista de 100 video_ids de YouTube a partir de una consulta de búsqueda
    Estos vídeos cumplen con el filtro de formato de duración media y subtítulos presentes
    '''
    
    url = "https://www.youtube.com/results"
    params = {"search_query": query,
              "sp": "EgQQASgB" #Filtro para videos formato media duración con subtítulos
              }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Buscar el script que contiene ytInitialData
    scripts = soup.find_all("script")

    if not scripts:
        print("got no scripts for query", query)

    for script in scripts:
        if "ytInitialData" in script.text:
            json_text = re.search(r"ytInitialData\s*=\s*(\{.*\});", script.text)
            if json_text:
                data = json.loads(json_text.group(1))
                break
    else:
        print("got no ytInitialData for query", query)
        return []

    # Buscar todos los videoId dentro del JSON
    video_ids = set()
    def extract_ids(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "videoId":
                    video_ids.add(v)
                else:
                    extract_ids(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_ids(item)

    extract_ids(data)

    return list(video_ids)

def get_random_ids(num_ids=25, after_date=None, before_date=None):
    '''
    Devuelve una lista de "num_ids" video_ids aleatorios a partir de palabras aleatorias y la función get_video_ids.
    Por cada palabra aleatoria se realiza una búsqueda en Youtube de videos que contengan esa palabra en el título y que hayan sido publicados entre "after_date" y "before_date".

    input:
    - num_ids
    '''
    ##Habría que añadir algo que controle que la fecha de inicio no sea posterior a la de fin, o que no se introduzcan fechas futuras, etc. 
    # Porque si no se mete en un bucle infinito.
    lista_palabras_aleatorias = []
    lista_ids_aleatorios = []
    w = RandomWord()
    pbar = tqdm(total=num_ids)
    try:
        while len(lista_ids_aleatorios) < num_ids:
            try: 
                random_word = w.word()
                query = f'\"{random_word}\" intitle:{random_word}'
                if after_date:
                    query += f' after:' + str(after_date)
                if before_date:
                    query += f' before:' + str(before_date)
                #print("running query", query)
                lista_ids_aleatorios.append(random.choice(get_video_ids(query)))
                lista_palabras_aleatorias.append(random_word)
                time.sleep(0.2) #para no hacer muchas queries seguidas
                pbar.update(1)
            except Exception as e: 
                pass #print("Ran into exception", e, "for word", random_word) #happens quite often when no videos found for a word in the last day     
    except KeyboardInterrupt:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"data/lista_ids_adultos{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(lista_ids_aleatorios, f)
            print("Ejecucion finalizada, guardada la lista de ids en local")
        raise KeyboardInterrupt
    finally:
        pbar.close()
    return lista_palabras_aleatorias, lista_ids_aleatorios

def get_random_ids_kids(num_ids=25, after_date=None, before_date=None, rango="0-4"):
    '''
    Devuelve una lista de "num_ids" video_ids aleatorios a partir de palabras aleatorias y la función get_video_ids.
    Por cada palabra aleatoria se realiza una búsqueda en Youtube de videos que contengan esa palabra en el título y que hayan sido publicados entre "after_date" y "before_date".
    '''
    ##Habría que añadir algo que controle que la fecha de inicio no sea posterior a la de fin, o que no se introduzcan fechas futuras, etc. 
    # Porque si no se mete en un bucle infinito.
    lista_palabras_aleatorias = []
    lista_ids_aleatorios = []
    while len(lista_ids_aleatorios)<num_ids:
        try: lista_palabras_aleatorias, lista_ids_aleatorios = get_vkids_ids(rango=rango,num_random_ids=num_ids)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except: pass

    return lista_palabras_aleatorias, lista_ids_aleatorios

# pbar = tqdm(total=num_ids)
#    while len(lista_ids_aleatorios) < num_ids:
#       try: 
#           random_word = w.word()
#           query = f'\"{random_word}\" intitle:{random_word}'
#           # if after_date:
#           #     query += f' after:' + str(after_date)
#           # if before_date:
#           #     query += f' before:' + str(before_date)
#          #print("running query", query)
#           lista_ids_aleatorios.append(random.choice(get_vkids_ids(query, rango)))
#           lista_palabras_aleatorias.append(random_word)
#           time.sleep(0.2) #para no hacer muchas queries seguidas
#           pbar.update(1)
#       except Exception as e: pass #print("Ran into exception", e, "for word", random_word) #happens quite often when no videos found for a word in the last day
