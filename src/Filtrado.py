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

def filtrado(df): 
    numero_pre_filtrado = len(df)
    max = df["Duracion"].quantile(0.95)
    min = df["Duracion"].quantile(0.05)
    bool_duracion = df[(df["duracion"] < min) | (df["duracion"] > max)].index
    valores = ["Descripcion","Tags", "Subtitulos"]
    for col in valores
    bool = df[df["Descripcion"].str.lower().eq("none")]
    bool
    
    numero_pos_filtrado = len(df)
    diff = numero_pre_filtrado - numero_pos_filtrado
    print(f'Partiendo de {numero_pre_filtrado}, se han eliminado {diff}, resultando en: {numero_pos_filtrado} columnas')
