import pandas as pd
from datetime import datetime
import json
import os
import tempfile
from Server_PD import get_minio_client
from Server_PD import upload_dataframe_minio

def unir_parquets_minio(bucket: str, prefix: str, claves: dict) -> pd.DataFrame:

    client = get_minio_client(claves)

    objects = client.list_objects(
        bucket_name=bucket,
        prefix=prefix,
        recursive=True
    )

    dfs = []

    for obj in objects:
        if obj.object_name.endswith(".parquet") and "union_dfs" not in obj.object_name:

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                temp_path = tmp.name

            client.fget_object(
                bucket_name=bucket,
                object_name=obj.object_name,
                file_path=temp_path
            )

            df = pd.read_parquet(temp_path)
            dfs.append(df)

            os.remove(temp_path)

    if not dfs:
        raise ValueError("No se encontraron archivos parquet.")

    df_final = pd.concat(dfs, ignore_index=True)

    print("Columnas encontradas:")
    print(df_final.columns)

    return df_final

def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    print("Filas originales:", len(df))

    print("\n--- VALORES NULOS POR COLUMNA ---")
    print(df.isnull().sum())

    # 1️⃣ Eliminar filas completamente vacías
    df = df.dropna(how="all")
    print("Después de eliminar filas totalmente vacías:", len(df))

    # 2️⃣ Eliminar duplicados por ID
    if "ID" in df.columns:
        df = df.drop_duplicates(subset=["ID"])
    print("Después de eliminar duplicados por ID:", len(df))

    # 3️⃣ Convertir columnas numéricas sin eliminar filas
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

    # 4️⃣ Convertir fecha
    if "Fecha_publicacion" in df.columns:
        df["Fecha_publicacion"] = pd.to_datetime(
            df["Fecha_publicacion"],
            errors="coerce"
        )

    # 5️⃣ Eliminar SOLO registros sin información textual clave REAL
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

    with open("Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)

    bucket = "pd1"
    prefix = "grupo1/"  # cambia si es necesario

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

    subir_union(df_limpio, bucket, prefix, claves)

    print("Proceso finalizado correctamente")