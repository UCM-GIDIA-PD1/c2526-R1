import time
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import json
import datetime

from Server_PD import download_dataframe_minio

with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
    claves = json.load(archivo)


    df = download_dataframe_minio("pd1", "grupo1/df_videos_2026-02-25_17-33-15",claves=claves,file_format="parquet")
    print(df.columns)
    #print(df[df["Made for kids"] == True].head())