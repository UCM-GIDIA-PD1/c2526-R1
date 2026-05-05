import io
import pandas as pd
from tqdm import tqdm
import easyocr
from Server_PD import get_minio_client, upload_dataframe_minio
import json

def extraer_texto_miniaturas(bucket, prefix, claves):
    client = get_minio_client(claves) 
    reader = easyocr.Reader(['en'])     # Idioma
    
    objects = client.list_objects(bucket, prefix=prefix, recursive=True) #
    data_ocr = []

    print("Extrayendo texto de las miniaturas (OCR)...")
    for obj in tqdm(list(objects)):
        if obj.object_name.endswith(".jpg"):
            # Descargar imagen de MinIO
            response = client.get_object(bucket, obj.object_name)
            img_data = response.read()
            
            # Leer texto de la imagen
            result = reader.readtext(img_data, detail=0) # detail=0 devuelve solo el texto
            texto_extraido = " ".join(result)
            
            video_id = obj.object_name.split("/")[-1].replace(".jpg", "")
            data_ocr.append({"ID": video_id, "OCR_text": texto_extraido})

    df_ocr = pd.DataFrame(data_ocr)
    return df_ocr

if __name__ == "__main__":
    with open("src/Private/claves.json", "r") as f:
        claves = json.load(f)
    
    df_ocr_final = extraer_texto_miniaturas("pd1", "grupo1/test_images/thumbnails/", claves)
    
    # Subimos el resultado a MinIO
    upload_dataframe_minio(df_ocr_final, "pd1", "grupo1/test_images/ocr_results", claves, "parquet")
    print("Proceso OCR finalizado y subido a MinIO.")