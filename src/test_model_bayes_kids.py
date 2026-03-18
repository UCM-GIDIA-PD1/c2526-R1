import joblib
import pandas as pd

# cargar modelo
model = joblib.load("nb_kids_model.joblib")

# ejemplo de datos
data = pd.DataFrame({
    "Titulo": ["Peppa Pig full episode"],
    "Descripcion": ["Fun cartoon adventure for kids"],
    "Tags": ["cartoon kids family"],
    "Subtitulos": ["lets play in the mud puddle"],
    "Generos": ["Animation"]
})

pred = model.predict(data)

print("Prediction:", pred)