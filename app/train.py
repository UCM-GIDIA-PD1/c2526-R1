from extraccion.collect_all_data import get_info
import re
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import LabelEncoder
import comun.training_utils as train
from comun.filter_and_divide_data import iso_a_minutos

class model_kids:
    def __init__(self, model, pipe):
        """
        Parametros entrada -> 
        model: Modelo que vamos a utilizar para la predicción.
        pipe: Pipeline de transformación de datos que vamos a utilizar

        Parametros salida -> 
            Predicción (string)

        Extrae los datos de un video a partir de la url y predice el resultado
        """
        self.model = model
        self.pipe = pipe
        self.answer = None
    def _get_data_and_predict(self, url: str):
        """
        Parametros entrada -> 
        model: Modelo correspondiente a la predicción que queramos hacer. (Generos o kids).
        url (string): Url del video que se quiere clasficar.

        Parametros salida -> 
            Predicción (string)

        Extrae los datos de un video a partir de la url y predice el resultado
        """
        # Extracción de ID
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id_match:
            return None
        
        video_id = video_id_match.group(1)

        # Extraemos los datos del video a partir del ID
        df_video = get_info(video_id)
        if df_video is None or df_video.empty:
            return None, 0.0
        print(df_video)

        # Predicción
        df_video["Duracion"] = df_video["Duracion"].apply(iso_a_minutos)        
        df_video_trans = self.pipe.transform(df_video)
        prediction= self.model.predict(df_video_trans)
        
        return prediction

class model_genres: 
    def __init__(self, model, label):
        self.model = model
        self.answer = None
        self.label = label
    def _get_data_and_predict(self, url: str):
        """
        Parametros entrada -> 
        model: Modelo correspondiente a la predicción que queramos hacer. (Generos o kids).
        url (string): Url del video que se quiere clasficar.

        Parametros salida -> 
            Predicción (string)

        Extrae los datos de un video a partir de la url y predice el resultado
        """
        # Extracción de ID
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id_match:
            return None, 0.0
        
        video_id = video_id_match.group(1)

        # Extraemos los datos del video a partir del ID
        df_video = get_info(video_id)
        
        if df_video is None or df_video.empty:
            return None, 0.0

        # Predicción
        prediction= self.model.predict(df_video)[0]

        try:
            prob = self.model.predict_proba(df_video).max()
        except:
            prob = 0.95
        
        return prediction, prob
