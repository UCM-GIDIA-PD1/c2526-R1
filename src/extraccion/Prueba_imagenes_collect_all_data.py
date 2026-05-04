import time
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import json
import datetime

from get_video_info_api import get_info
from get_video_ids_bs4 import get_random_ids, get_random_ids_kids
from Server_PD import upload_dataframe_minio

# Redes neuronales
import requests                                 # Descargar la imagen de URL
from Server_PD import upload_image_minio 

def collect_all_data(num_videos=500, fecha=None, proporcion_adults=0.8):
    '''
    Elegimos y descargamos n videos random, donde n es el parámetro num_videos de la función. 
    Dividimos los datos, para que aproximadamente 80% de los videos recopilados sean de adultos y 20% de niños 
    Los videos para niños se dividen de manera equitativa entre videos para niños de 0-4 años, 5-8 años y 9-12 años.
    '''
    try:
        with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
            claves = json.load(archivo)
    except FileNotFoundError:
        print("Error: No se encontró el archivo de claves.")
        return None
        
    num_adults = int(num_videos * proporcion_adults)
    num_kids = num_videos - num_adults
    num_rango_kids = [num_kids // 3, num_kids // 3, num_kids - 2 * (num_kids // 3)]

    print("PART 1.1 - GETTING RANDOM IDS FOR ADULTS VIDEOS")
    palabras, ids = get_random_ids(num_ids=num_adults, after_date=str(datetime.date.today()-datetime.timedelta(days=1)))
    print(list(zip(palabras,ids)))

    ids_ages = ids
    for i in range(len(ids_ages)):
        ids_ages[i] = [ids_ages[i], "Adult"]


    print("PART 1.2 - GETTING RANDOM IDS FOR KIDS VIDEOS")
    for i, rango in enumerate(["0-4", "5-8", "9-12"]):
        try:
            palabras_kids, ids_kids = get_random_ids_kids(num_ids=num_rango_kids[i], after_date=str(datetime.date.today()-datetime.timedelta(days=1)), rango=rango)
            print('ola')
        except KeyboardInterrupt:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"data/lista_ids_ages{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(ids_ages, f)
                print("Ejecucion finalizada, guardada la lista de ids en local")
            raise KeyboardInterrupt
        print(list(zip(palabras_kids,ids_kids)))
        palabras.extend(palabras_kids)
        for id in ids_kids:
            ids_ages.append([id, rango])
    try:
        print("PART 2 - PROCESSING VIDEOS")
        df_videos = []

        for id in tqdm(ids_ages):
            try:
                row = get_info(id[0]) # row es un DataFrame de una fila
                if row is None: continue

                # Acceso correcto al valor de la URL en el DataFrame
                img_url = row["Thumbnail_url"].iloc[0] 
                
                if img_url and img_url != "None":
                    img_res = requests.get(img_url, timeout=10)
                    if img_res.status_code == 200:
                        # Ahora 'claves' ya existe aquí
                        upload_image_minio(
                            image_bytes=img_res.content,
                            bucket="pd1",
                            object_name=f"grupo1/test_images/thumbnails/{id[0]}.jpg",
                            claves=claves
                        )
                
                # Asignación de edad al DataFrame y guardado
                row["Rango_edad"] = id[1]
                df_videos.append(row)
                time.sleep(0.2)
            except Exception as e: 
                print(f"Error en video {id[0]}: {e}")
    finally:
        # Consolidamos los datos extraídos
        df_data = pd.concat(df_videos, ignore_index=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Guardar en MinIO usando las claves ya cargadas al inicio
        try:
            upload_dataframe_minio(
                df=df_data, 
                bucket="pd1", 
                object_name=f"grupo1/test_images/raw/df_videos_PRUEBA_{timestamp}", 
                claves=claves, # Usamos la variable ya definida
                file_format="parquet"
            )  
            print(f"Datos de prueba subidos correctamente: df_videos_PRUEBA_{timestamp}")
        except Exception as e:
            print(f"No se pudo subir a MinIO, guardando en local. Error: {e}")
            DataFrame(df_data).to_parquet(path=f"src/data/df_videos_{timestamp}.parquet", index=False)
            
        return df_data

