from collect_all_data import get_info
import re
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import LabelEncoder
import training_utils as train
from filter_and_divide_data import iso_a_minutos
import string

class model_kids:
    def __init__(self, model, pipe):
        """
        Inicializa la clase

        Parameters
        ----------
        model: Modelo clasificador
            Modelo que vamos a utilizar para la predicción de made for kids

        pipe: Pipeline que incluya transformaciones de texto y dimensionalidad
            Pipeline de transformación que vamos a utilizar

        Returns
        -------
        None
            
        """
        self.model = model
        self.pipe = pipe
    def _get_data_and_predict(self, url: str) -> bool: 
        """
        Obtiene la información de la url de un vídeo y predice si es apto o no para niños (True o False)

        Parameters
        ----------
        url: string
            Url de youtube completa para acceder al vídeo 

        Returns
        -------
        Prediction: bool
            True: El video es apto para niños 
            
            False: El video no es apto para niños
            
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
    def __init__(self, model, pipe, label):
        """
        Inicializa la clase

        Parameters
        ----------
        model: Modelo clasificador
            Modelo que vamos a utilizar para la predicción de Generos

        pipe: Pipeline que incluya transformaciones de texto y dimensionalidad
            Pipeline de transformación que vamos a utilizar
        
        label: Label enconder
            Label enconder para obtener el nombre de las clases

        Returns
        -------
        None
            
        """
        self.model = model
        self.pipe = pipe
        self.label = label
    def _get_data_and_predict(self, url: str) -> string:
        """
        Obtiene la información de la url de un vídeo y predice su Genero (string)

        Parameters
        ----------
        url: string
            Url de youtube completa para acceder al vídeo 

        Returns
        -------
        Prediction: String
            Posible clase entre las 10 existentes
            
        """
        #Extracción de ID
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
        prediction_trans = self.transform_label(prediction=prediction)
        
        
        return prediction_trans
    
    def transform_label(self, prediction: int) -> string: 
        """
        Recibe una prediccion numerica y la transforma en su clase (string) correspondiente

        Parameters
        ----------
        Prediction: int
            Predicción con valor númerico 

        Returns
        -------
        Prediction: String
            Posible clase entre las 10 existentes
            
        """
        return self.label.inverse_transform(prediction)[0]
        
