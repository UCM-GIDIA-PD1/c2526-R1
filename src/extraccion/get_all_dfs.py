import pandas as pd
from datetime import datetime
import json
import os
import tempfile
from Server_PD import get_minio_client
from Server_PD import upload_dataframe_minio
#   bucket = "pd1"
#   prefix = "grupo1/"
#   with open("Private/claves.json", "r", encoding="utf-8") as archivo:
#   claves = json.load(archivo)
def unir_parquets_minio(bucket: str, prefix: str, claves: dict) -> pd.DataFrame:

    client = get_minio_client(claves) #Iniciamos el server

    objects = client.list_objects( #Obtenemos todos los objetos
        bucket_name=bucket,
        prefix=prefix,
        recursive=True
    )

    dfs = [] #Creamos la lista de datasets

    for obj in objects:
        if obj.object_name.endswith(".parquet") and "union_dfs" not in obj.object_name:

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                temp_path = tmp.name

            client.fget_object( #Accedemos al objeto
                bucket_name=bucket,
                object_name=obj.object_name,
                file_path=temp_path
            )

            df = pd.read_parquet(temp_path)
            dfs.append(df) #Unimos el df a la lista de df

            os.remove(temp_path)

    if not dfs: #Si no hay archivos parquet lanzamos error
        raise ValueError("No se encontraron archivos parquet.")

    df_final = pd.concat(dfs, ignore_index=True) #Unimos los dfs en uno final

    print("Columnas encontradas:")
    print(df_final.columns)

    return df_final #Devolvemos la unión

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    print("Filas originales:", len(df))

    print("\n--- VALORES NULOS POR COLUMNA ---")
    print(df.isnull().sum())

    print("\n--- CONTEO DE 'None' o 'none' o (vacío) EN COLUMNAS STRING ---")
    columnas_string = df.select_dtypes(include=["object", "string"]).columns

    for col in columnas_string:
        conteo_none = df[col].astype(str).str.lower().eq("none").sum() + df[col].astype(str).eq("").sum()
        print(f"{col}: {conteo_none}")

    #Elimina filas completamente vacías
    df = df.dropna(how="all")
    print("Después de eliminar filas totalmente vacías:", len(df))

    #Elimina duplicados por ID
    if "ID" in df.columns:
        df = df.drop_duplicates(subset=["ID"])
    print("Después de eliminar duplicados por ID:", len(df))

    #Convierte columnas numéricas sin eliminar filas
    for col in ["Visualizaciones", "Numero_likes"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    #Convierte fecha
    if "Fecha_publicacion" in df.columns:
        df["Fecha_publicacion"] = pd.to_datetime(
            df["Fecha_publicacion"],
            errors="coerce"
        )

    # Elimina SOLO registros sin información textual clave REAL
    columnas_clave = ["Titulo", "Descripcion"]

    columnas_existentes = [c for c in columnas_clave if c in df.columns]

    if columnas_existentes:
        df = df.dropna(subset=columnas_existentes)

    print("Filas después de limpieza:", len(df))
    print("Tipos finales:\n", df.dtypes)

    return df

def subir_union(df_final: pd.DataFrame, bucket: str, prefix: str, claves: dict):

    fecha_actual = datetime.now().strftime("%Y%m%d")

    nombre_archivo = f"{prefix}union_dfs_{fecha_actual}"

    upload_dataframe_minio(
    df=df_final,
    bucket=bucket,
    object_name=nombre_archivo,
    claves=claves,
    file_format="parquet"
)

    print("Archivo subido correctamente como:", nombre_archivo)


"""
#Analiza dfs y los limpi
if __name__ == "__main__":

    with open("Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)

    bucket = "pd1"
    prefix = "grupo1/"

    df_final = unir_parquets_minio(bucket, prefix, claves)

    print("\n--- ANÁLISIS DE DUPLICADOS ---")

    total_filas = len(df_final)
    ids_unicos = df_final["ID"].nunique()

    print("Total filas:", total_filas)
    print("IDs únicos:", ids_unicos)
    print("Duplicados encontrados:", total_filas - ids_unicos)

    print("\nTop 5 IDs más repetidos:")
    print(df_final["ID"].value_counts().head())

    print("\n--- LIMPIEZA ---")

    df_limpio = limpiar_dataframe(df_final)

    print("\nFilas eliminadas en limpieza:", total_filas - len(df_limpio))

    print("\nProceso finalizado (modo prueba, no se sube a MinIO)")
"""

#Para unir los dfs y subirlo al minio
if __name__ == "__main__":

    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)

    bucket = "pd1"
    prefix = "grupo1/raw/"

    df_final = unir_parquets_minio(bucket, prefix, claves)

    print("\n--- ANÁLISIS DE DUPLICADOS ---")

    total_filas = len(df_final)
    ids_unicos = df_final["ID"].nunique()

    print("Total filas:", total_filas)
    print("IDs únicos:", ids_unicos)
    print("Duplicados encontrados:", total_filas - ids_unicos)

    print("\nTop 5 IDs más repetidos:")
    print(df_final["ID"].value_counts().head())

    print("\n--- LIMPIEZA ---")

    df_limpio = limpiar_dataframe(df_final)

    print("\nFilas eliminadas en limpieza:", total_filas - len(df_limpio))
    prefix = "grupo1/clean/"
    subir_union(df_limpio, bucket, prefix, claves)

    print("Proceso finalizado correctamente")