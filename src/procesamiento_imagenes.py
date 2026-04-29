import io
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from Server_PD import get_minio_client, upload_dataframe_minio

def generar_embeddings_imagenes(bucket, prefix, claves):
    client = get_minio_client(claves) 

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    objects = client.list_objects(bucket, prefix=prefix, recursive=True) 
    data_embeddings = []

    print("Generando embeddings de imágenes...")
    for obj in tqdm(list(objects)):
        if obj.object_name.endswith(".jpg"):
            # Descargar imagen de MinIO a memoria
            response = client.get_object(bucket, obj.object_name)
            img_data = response.read()
            
            #  Preprocesamiento 
            img = image.load_img(io.BytesIO(img_data), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Vector
            preds = model.predict(x, verbose=0)
            
            # El ID es el nombre del archivo sin el .jpg
            video_id = obj.object_name.split("/")[-1].replace(".jpg", "")
            data_embeddings.append({"ID": video_id, "img_embedding": preds.flatten()})

    df_emb = pd.DataFrame(data_embeddings)
    return df_emb

if __name__ == "__main__":
    import json
    with open("src/Private/claves.json", "r") as f:
        claves = json.load(f)
    
    df_vectores = generar_embeddings_imagenes("pd1", "grupo1/test_images/thumbnails/", claves)
    
    # Subimos el resultado a MinIO
    upload_dataframe_minio(df_vectores, "pd1", "grupo1/test_images/embeddings_vision", claves, "parquet")