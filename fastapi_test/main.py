'''
Ejemplo de servidor web usando FastAPI.

En modo desarrollo se ejecuta con:
> uv run fastapi dev

En modo despliegue se ejecuta con:
> uv run fastapi run

Por defecto escucha en http://127.0.0.1:8000
Documentación interactiva en http://127.0.0.1:8000/docs
'''

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import Annotated
from pydantic import BaseModel
from joblib import load

# Se ejecuta una vez al arrancar el servidor (startup) y al apagarlo (shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: cargar el modelo en memoria para que esté disponible en las peticiones.
    app.state.model = load('models/final_model.joblib')
    yield  # El servidor atiende peticiones mientras está aquí
    # Shutdown: aquí se liberarían recursos si fuera necesario

# Crear la aplicación web
app = FastAPI(lifespan=lifespan)

# Página de inicio con enlaces a todos los ejemplos
@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, name='index.html')


# -----------------------------------------------------------------------------
# BLOQUE 1: Peticiones HTTP básicas
#
# Los dos tipos de petición más comunes son GET y POST.
# En ambos casos el cliente envía datos al servidor y el servidor responde con JSON.
#
# Se pueden probar desde la documentación interactiva en http://127.0.0.1:8000/docs
# -----------------------------------------------------------------------------

# Petición GET: los parámetros viajan en la propia URL, visibles para todos.
# Ejemplo: http://127.0.0.1:8000/echo?a=Hola&b=1
@app.get('/echo')
def echo_get(a: str, b: int):
    print(f'echo_get a:{a} b:{b}')
    return {'a': a, 'b': b}

# Petición POST: los parámetros viajan en el cuerpo (body) de la petición, no en la URL.
# Permite enviar más información y de forma más segura (p.ej. contraseñas).
@app.post('/echo')
def echo_post(a: str, b: int):
    print(f'echo_post a:{a} b:{b}')
    return {'a': a, 'b': b}


# -----------------------------------------------------------------------------
# BLOQUE 2: Servidor web con páginas HTML
#
# El servidor puede devolver páginas HTML además de JSON.
# Hay dos formas: servir ficheros HTML estáticos o usar plantillas (templates).
# -----------------------------------------------------------------------------

# Ficheros HTML estáticos: se sirven tal cual, sin modificar.
# Por ejemplo: http://127.0.0.1:8000/static/testget.html
app.mount('/static', StaticFiles(directory='static', html=True), name='static')

# Templates (plantillas): ficheros HTML con variables que el servidor rellena antes de enviar.
templates = Jinja2Templates(directory="templates")

# Petición GET desde un formulario HTML con respuesta en HTML.
# El formulario está en http://127.0.0.1:8000/static/testget.html
# El parámetro 'request' es necesario para que FastAPI pueda construir la respuesta HTML.
@app.get('/echo/html', response_class=HTMLResponse)
def echo_get_html(request: Request, a: str, b: int):
    print(f'echo_get_html a:{a} b:{b}')
    return templates.TemplateResponse(request, name='testres.html', context={'a': a, 'b': b})

# Petición POST desde un formulario HTML con respuesta en HTML.
# El formulario está en http://127.0.0.1:8000/static/testpost.html
# Cuando los datos vienen de un formulario POST hay que indicarlo con Annotated[..., Form()].
@app.post('/echo/html', response_class=HTMLResponse)
def echo_post_html(request: Request,
                   a: Annotated[str, Form()],
                   b: Annotated[int, Form()]):
    print(f'echo_post_html a:{a} b:{b}')
    return templates.TemplateResponse(request, name='testres.html', context={'a': a, 'b': b})


# -----------------------------------------------------------------------------
# BLOQUE 3: Despliegue de un modelo de Machine Learning
#
# El modelo se entrena una sola vez con train.py y se guarda en models/final_model.joblib.
# El servidor lo carga al arrancar y lo usa para responder peticiones de predicción.
# Se ofrecen dos interfaces: una JSON (para integrar con otros sistemas) y una HTML (para el navegador).
# -----------------------------------------------------------------------------

# Clases que definen la estructura de los datos de entrada y salida.
# Pydantic valida automáticamente que los datos recibidos tienen el tipo correcto.
class Flower(BaseModel):
    '''Medidas de una flor del dataset Iris (en centímetros).'''
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    '''Resultado de la predicción.'''
    label: int          # Clase predicha (0=setosa, 1=versicolor, 2=virginica)
    probs: list[float]  # Probabilidad de cada clase

def _predict_flower_class(flower: Flower):
    '''Devuelve la clase predicha y las probabilidades de cada clase.'''
    # Convertir los campos del objeto a una lista de valores numéricos
    sample = list(vars(flower).values())
    # equivalente a: sample = [flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width]

    y_pred = app.state.model.predict([sample])[0]
    probs = app.state.model.predict_proba([sample])[0]

    return y_pred, probs

# Interfaz JSON: recibe los datos de la flor en formato JSON y devuelve la predicción en JSON.
# Útil para integrar el modelo con otras aplicaciones o lenguajes de programación.
@app.post("/iris/predict", response_model=Prediction)
def predict_iris(flower: Flower):
    print(f'predict_iris flower:{flower}')
    y_pred, probs = _predict_flower_class(flower)
    return Prediction(label=y_pred, probs=probs)

# Interfaz HTML: recibe los datos desde un formulario web y devuelve la predicción en HTML.
# El formulario está en http://127.0.0.1:8000/static/flower1.html
@app.post("/iris/predict/html", response_class=HTMLResponse)
def predict_iris_html(request: Request,
                      flower: Annotated[Flower, Form()]):
    print(f'predict_iris_html flower:{flower}')
    y_pred, probs = _predict_flower_class(flower)
    context = {
        'clase': ['setosa', 'versicolor', 'virginica'][y_pred],
        'prob_setosa': probs[0],
        'prob_versicolor': probs[1],
        'prob_virginica': probs[2],
    }
    return templates.TemplateResponse(request, name='response.html', context=context)
