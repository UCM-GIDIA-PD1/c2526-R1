# Archivo para importar las claves privadas

import json

def get_key():
    with open("Private/Claves.json", "r", encoding="utf-8") as archivo:
        keys = json.load(archivo)
    return keys