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

    #Para filtrar
def informacion_vacia(df): 
    dicc = {}
    print("\n--- CONTEO DE VALORES NULOS EN COLUMNAS STRING ---")
    columnas_string = df.select_dtypes(include=["object", "string"]).columns
    columnas_no_string = df.select_dtypes(exclude=["object", "string"]).columns


    for col in columnas_string:
        conteo_none = df[col].astype(str).str.lower().eq("none").sum() + df[col].astype(str).eq("").sum()
        print(f"{col}: {conteo_none}")
        dicc[col] = conteo_none
        
    print("\n--- CONTEO DE VALORES NULOS EN COLUMNAS NO STRING ---")

    for col in columnas_no_string:
        conteo_null = df[col].isna().sum()
        print(f"{col}: {conteo_null}")
        dicc[col] = conteo_null
    return dicc

def filtrado(df_original):
    df = df_original.copy()
    numero_pre_filtrado = len(df)
    max = df["Duracion"].quantile(0.95)
    min = df["Duracion"].quantile(0.05)
    bool_duracion = df[(df["duracion"] < min) | (df["duracion"] > max)].index
    valores = ["Descripcion","Tags", "Subtitulos"]
    bool_text = df[df["Titulo"] != ""] #Filtro completamente positivo
    for col in valores:
        bool1 = df[df[col].str.lower().eq("none")]
        bool2 = df[df[col].str.lower().eq("")]
        bool_text = bool_text or (bool1 and bool2)   
    bool_final = bool_duracion and bool_text
    df = df[bool_final]
    numero_pos_filtrado = len(df)
    diff = numero_pre_filtrado - numero_pos_filtrado
    print(f'Partiendo de {numero_pre_filtrado}, se han eliminado {diff}, resultando en: {numero_pos_filtrado} columnas')
    return df
