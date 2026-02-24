import time
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import json
import datetime

from get_video_info_api import get_info
from get_video_ids_bs4 import get_random_ids, get_random_ids_kids
from Server_PD import upload_dataframe_minio

def collect_all_data(num_videos):
    '''
    Elegimos y descargamos n videos random, donde n es el parámetro num_videos de la función. 
    Dividimos los datos, para que aproximadamente 80% de los videos recopilados sean de adultos y 20% de niños 
    Los videos para niños se dividen de manera equitativa entre videos para niños de 0-4 años, 5-8 años y 9-12 años.
    '''

    num_adults = int(num_videos * 0.8)
    num_kids = num_videos - num_adults
    num_rango_kids = [num_kids // 3, num_kids // 3, num_kids - 2 * (num_kids // 3)]

    print("PART 1.1 - GETTING RANDOM IDS FOR ADULTS VIDEOS")
    palabras, ids = get_random_ids(num_ids=num_adults, after_date=str(datetime.date.today()-datetime.timedelta(days=1)))
    print(list(zip(palabras,ids)))

    print("PART 1.2 - GETTING RANDOM IDS FOR KIDS VIDEOS")
    for i, rango in enumerate(["0-4", "5-8", "9-12"]):
        palabras_kids, ids_kids = get_random_ids_kids(num_ids=num_rango_kids[i], after_date=str(datetime.date.today()-datetime.timedelta(days=1)), rango=rango)
        print(list(zip(palabras_kids,ids_kids)))
        palabras.extend(palabras_kids)
        ids.extend(ids_kids)

    print("PART 2 - PROCESSING VIDEOS")
    df_videos = []

    for id in tqdm(ids): #tqdm
        try:
            df_videos.append(get_info(id))
            #print(df_videos)
            time.sleep(0.2) #to not get too many requests error
        except Exception as e: print("Ran into exception", e, "for video", id) #for some videos the downloader does not work for some reason

    df_data = pd.concat(df_videos, ignore_index=True)

    #Guardar en local en formato .parquet con la fecha
    #data_parquet = df_data.to_parquet(f"src/data/data_videos_{timestamp}.parquet", index=False)

    #Guardar en MinIO
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    upload_dataframe_minio(df = df_data, bucket = "pd1/grupo1/", object_name=f"df_videos_{timestamp}", claves=claves, file_format="parquet") 

    return df_data

if __name__ == '__main__':
    # data = collect_all_data(20)#(1000)

    #Para probar el trabajo con MinIO
    data = pd.read_csv("src\data\data_videos_2026-02-24_14-06-23.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    upload_dataframe_minio(df = data, bucket = "pd1/grupo1/", object_name=f"df_videos_{timestamp}", claves=claves, file_format="parquet")
