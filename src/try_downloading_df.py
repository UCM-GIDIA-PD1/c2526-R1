import time
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
import json
from get_video_info_api import get_info
import datetime

from Server_PD import download_dataframe_minio

with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
    claves = json.load(archivo)


    df = download_dataframe_minio("pd1", "grupo1/clean/union_dfs_20260309",claves=claves,file_format="parquet")
    for i in df["ID"]: 
        df = get_info(i)
        print(df["ContentRating"])
    #print(df[df["Made for kids"] == True].head())