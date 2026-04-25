#Incluye todas las funcionas para hacer carga, descarga e inicializar en minio
#Sirve tanto para csv como para parquets
import joblib
import tempfile
import os
import io
import pandas as pd
from minio import Minio

def get_minio_client(claves):
    """
    Crea y devuelve un cliente de MinIO usando variables de entorno.
    """
    return Minio(
        endpoint=claves["Url"],
        access_key=claves["Access_Key"],
        secret_key=claves["Secret_Key"],
        secure=True
    )

#Subir el dataframe
def upload_dataframe_minio(
    df: pd.DataFrame, #El dataframe a utilizar
    bucket: str,      #Nombre de donde lo vamos a guardar "pd1/grupo1/"
    object_name: str, #Nombre del objeto a guardar
    claves: dict,     #Diccionario con las claves
    file_format: str = "csv", #Formato
):
    """
    Sube un DataFrame a MinIO en formato CSV o Parquet.
    """
    client = get_minio_client(claves)

    suffix = ".csv" if file_format == "csv" else ".parquet"
    #Crea archivo temporal
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name

    # Guardar archivo temporal
    if file_format == "csv":
        df.to_csv(temp_path, index=False)
    elif file_format == "parquet":
        df.to_parquet(temp_path, index=False)
    else:
        raise ValueError("Formato no soportado. Usa 'csv' o 'parquet'.")

    # Subir a MinIO
    client.fput_object(
        bucket_name=bucket,
        object_name=object_name + suffix,
        file_path=temp_path
    )
    #Borra archivo temporal
    os.remove(temp_path)

#Descargar los datos y devolver un dataframe
def download_dataframe_minio(
    bucket: str, #Dirección que queremos descargar, "pd1"
    object_name: str, #Nombre del archivo que queremos ponerle
    claves:dict, #Diccionario con las claves
    file_format: str = "csv" #Formato
) -> pd.DataFrame:
    """
    Descarga un archivo desde MinIO y lo devuelve como DataFrame.
    """
    client = get_minio_client(claves)

    suffix = ".csv" if file_format == "csv" else ".parquet"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name

    #Obtiene el archivo 
    client.fget_object(
        bucket_name=bucket,
        object_name=object_name + suffix,
        file_path=temp_path
    )

    # Leer archivo
    if file_format == "csv":
        df = pd.read_csv(temp_path)
    elif file_format == "parquet":
        df = pd.read_parquet(temp_path)
    else:
        raise ValueError("Formato no soportado. Usa 'csv' o 'parquet'.")

    #Elimina el archivo temporal
    os.remove(temp_path)

    return df

def upload_model_minio(
    model,            
    bucket: str,     
    object_name: str, # Nombre del archivo
    claves: dict,     
):
    """
    Sube un modelo (objeto de ML) a MinIO en formato .joblib.
    """
    client = get_minio_client(claves)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        temp_path = tmp.name

    try:
        joblib.dump(model, temp_path)
        
        client.fput_object(
            bucket_name=bucket,
            object_name=f"{object_name}.joblib",
            file_path=temp_path
        )
        print(f"Modelo subido con éxito a: {bucket}/{object_name}.joblib")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  

def download_model_minio(          
    bucket: str,     
    object_name: str, # Nombre del archivo
    claves: dict,     
):
    """
    Descarga un modelo (objeto de ML) desde MinIO.
    """
    client = get_minio_client(claves)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Descargar archivo desde MinIO
        client.fget_object(
            bucket_name=bucket,
            object_name=f"{object_name}.joblib",
            file_path=temp_path
        )

        # Cargar modelo
        model = joblib.load(temp_path)

        print(f"Modelo descargado con éxito desde: {bucket}/{object_name}.joblib")

        return model

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Imagenes (Prueba para redes neuronales - Maria - 22/04)
def upload_image_minio(image_bytes, bucket, object_name, claves):
    """
    Sube el contenido binario de una imagen a MinIO.
    """
    client = get_minio_client(claves) # Reutiliza tu función
    
    # Convertimos los bytes en un flujo que MinIO pueda leer
    data = io.BytesIO(image_bytes)
    
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=data,
        length=len(image_bytes),
        content_type='image/jpeg'
    )