#Para filtrar los datos y dividirlos en entrenamiento, validación y test
from Server_PD import download_dataframe_minio, upload_dataframe_minio
import pandas as pd
import json

def divide_save_data(df):
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)

    df_train = df.sample(frac=0.7, random_state=1).reset_index()
    df_test = df.drop(df_train.index)
    df_val = df_test.sample(frac=0.5, random_state=1).reset_index()
    df_test = df.drop(df_val.index).reset_index()

    upload_dataframe_minio(df_train, "pd1", "grupo1/modelos/train", claves, "parquet")
    upload_dataframe_minio(df_test, "pd1", "grupo1/modelos/test", claves, "parquet")
    upload_dataframe_minio(df_val, "pd1", "grupo1/modelos/validation", claves, "parquet")

