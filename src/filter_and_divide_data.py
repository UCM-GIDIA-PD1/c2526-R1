#Para filtrar los datos y dividirlos en entrenamiento, validación y test
from comun.Server_PD import download_dataframe_minio, upload_dataframe_minio
import comun.analisisutils as utils
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def made_for_kids(df): 
    """
    Corrige los valores de la columna made for kids
    Si es apto para niños == True
    Si no es apto para niños == False

    Parameters
    ----------
    df:
        Dataframe a modificar

    Returns
    -------
    df:
        Datraframe modificado
    """ 
    df["Made for kids"] = df["Rango_edad"] != 'Adult'
    
    return df
def download_latest_extraction_correct(filtrado = False):
    """
    Descarga el último dataframe de extracción (se pone a mano)

    Parameters
    ----------
    Filtrado: bool
        Marca si se quiere filtrar el dataframe

    Returns
    -------
    df:
        Datraframe descargado
    """ 
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    
    df = download_dataframe_minio("pd1", "grupo1/clean/union_dfs_20260309", claves, "parquet") #Descargamos el más reciente
    if filtrado: 
        df = filtrado(df) #Filtradomos el df
    df = made_for_kids(df) # Corregimos los kids
    return df


def get_data_models_train_test(filtrado = False, to_predict = "Made for kids"):
    """
    Obten un X_train, y_train, X_test, y_test más reciente posible.
    Estratificado para niños o generos

    Parameters
    ----------
    Filtrado: bool
        Marca si se quiere utilizar datos filtrados o sin filtrar
    to_predict: string
        Dice que columna vamos a predecir: Generos o Made for kids

    Returns
    -------
    X_train, X_test, y_train, y_test:
        Datos descargados
    """ 
    df = download_latest_extraction_correct(filtrado).copy()
    y = df[to_predict]
    X = df.drop([to_predict], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,      
        random_state=42,    
        stratify=y          
    )
    
    return X_train, X_test, y_train, y_test


def informacion_vacia(df): 
    """
    Informa sobre los valores nulos o vacíos de un dataframe

    Parameters
    ----------
    df:
        Dataframe a buscar información

    Returns
    -------
    dicc:
        Diccionario con columna: numeros de filas vacías o nulas
    """ 
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
    """
    Filtra un dataframe informando sobre las filas eliminadas.
    Los criterios son eliminar todos los videos con duraciones extremas
    Los criterios 
    True significa que tiene que ser eliminado

    Parameters
    ----------
    df_original:
        Dataframe a filtrar

    Returns
    -------
    df_filtrado:
        Dataframe filtrado
    """ 
    import pandas as pd
    df = df_original.copy()

    # Convertir Duracion a minutos y asegurar tipo numérico
    df['Duracion'] = df['Duracion'].apply(utils.iso_a_minutos)
    numero_pre_filtrado = len(df)

    # Calcular percentiles
    max_val = df["Duracion"].quantile(0.95)
    min_val = df["Duracion"].quantile(0.05)

    # Mascara de duración (True si fuera de rango)
    bool_duracion = (df["Duracion"] < min_val) | (df["Duracion"] > max_val)
    print(f'Numero de videos con duraciones atípicas: {bool_duracion.sum()}')

    # Inicializamos la máscara de texto como True (filtrar después)
    bool_text = pd.Series(True, index=df.index)

    # Columnas a revisar
    valores = ["Descripcion", "Tags", "Subtitulos"]

    for col in valores:
        # Convertimos a string y lowercase para comparar
        col_lower = df[col].astype(str).str.lower()
        # True si es "none" o vacío
        col_invalid = (col_lower == "none") | (col_lower == "")
        # Combinamos con máscara general
        bool_text &= col_invalid  

    # Combinamos máscaras: duración fuera de rango OR texto inválido
    print(f'Numero de videos sin información textual: {bool_text.sum()}')
    bool_final = ~(bool_duracion | bool_text)  # seleccionamos los válidos

    df_filtrado = df[bool_final]

    #Ahora filtramos por generos poco representativos 
    first = len(df_filtrado)
    frecuencias = df_filtrado["Generos"].value_counts()
    generos_validos = frecuencias[frecuencias >= 500].index
    df_filtrado = df_filtrado[df_filtrado["Generos"].isin(generos_validos)]
    print(f'Numero de videos con poca representación de generos: {first - len(df_filtrado)}')

    numero_pos_filtrado = len(df_filtrado)
    diff = numero_pre_filtrado - numero_pos_filtrado
    df_filtrado = df_filtrado.reset_index(drop=True)
    print(f'Partiendo de {numero_pre_filtrado}, se han eliminado {diff}, resultando en: {numero_pos_filtrado} filas')


    return df_filtrado

def divide_save_data(df, name): #Deberíamos eliminarla
    """
    Divide un df en datos de train y test. 
    Los sube al minio con un name

    Parameters
    ----------
    df:
        Dataframe a dividir
    
    name: string
        Nombre del archivo

    Returns
    -------
    df:
        Dataframe filtrado
    """     
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)

    df_train = df.sample(frac=0.7, random_state=1)
    df_test = df.drop(df_train.index)
    df_val = df_test.sample(frac=0.5, random_state=1)
    df_test = df_test.drop(df_val.index)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    upload_dataframe_minio(df_train, "pd1", f"grupo1/modelos/train_{name}", claves, "parquet")
    upload_dataframe_minio(df_test, "pd1", f"grupo1/modelos/test_{name}", claves, "parquet")
    upload_dataframe_minio(df_val, "pd1", f"grupo1/modelos/validation_{name}", claves, "parquet")

if __name__ == '__main__':
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    
    X_train, X_test, y_train, y_test = get_data_models_train_test()
    print(X_train)
    print(y_train.value_counts())
    print(y_test.value_counts())
    #print(df_filtered.columns)
    #print(df_filtered["Made for kids"].value_counts())
    #print(df["Made for kids"].value_counts())
