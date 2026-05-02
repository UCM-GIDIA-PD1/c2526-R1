# **SAFE FOR KIDS (SFKids)**

![Imagen portada](./Images%20readme/imagen%20portada.png)

## **Índice**

1. [Descripción de los objetivos](#1-descripción-de-los-objetivos)

2. [Estructura del repositorio](#2-estructura-del-repositorio)

3. [Iniciar el entorno de desarrollo y sus dependecias](#3-️-iniciar-el-entorno-de-desarrollo-y-sus-dependencias)

4. [Ejecutar los scripts del proyecto](#4-ejecutar-los-scripts-del-proyecto)

5. [Mejores modelos construídos](#5-mejores-modelos-construídos)

6. [Instrucciones para ejecutar la aplicación web](#6-instrucciones-para-ejecutar-la-aplicación-web)

7. [Instrucciones para crear y ejecutar el contenedor](#7-instrucciones-para-crear-y-ejecutar-el-contenedor)

8. [Equipo de desarrollo](#8-equipo-de-desarrollo)

### **1. Descripción de los objetivos**

1. Desarrollar una herramienta capaz de determinar de manera autónoma si un video de Youtube va a estar destinado o no para niños. Para ello se usa un modelo de aprendizaje autónomo entrenado para que sea capaz de decidir, a partir de un umbral de decisión, si es apto o no.

2. Asignar un género principal a cada video según su temática general (e.g, Educación, Deportes, Videojuegos), permitiendo a cualquier usuario filtrar contenido de manera precisa según sus intereses o necesidades pedagógicas.

### **2. Estructura del repositorio**

**Archivos:**

- `.gitignore`: Contiene los archivos que no se deben subir al git desde el repositorio local.

- `.python-version`: Contiene la versión de python usada en el proyecto.

- `pyproyect.toml` y `uv.lock`: Contiene la configuración del entorno, con las dependencias y las versiones correspondientes a la versión de python.

**Carpetas:**

- Carpeta `Images readme`: Contiene las imágenes de este archivo.

- Carpeta `app`: Contiene los ficheros necesarios para ejecutar la aplicación web (API y Frontend), organizada de la siguiente manera:

    - `main.py`: Punto de entrada de la aplicación. Contiene la lógica del servidor, la definición de los endpoints de la API REST y la carga de los modelos entrenados para generar las predicciones en tiempo real.

    - `train.py`: Script que gestiona la inferencia; encapsula la lógica de extracción de ID mediante expresiones regulares, la obtención de metadatos vía API y la aplicación de transformaciones para devolver la clasificación final (booleana para niños o string para géneros).

    - Carpeta `templates`: Almacena las interfaces de usuario (HTML) que permiten la interacción con los modelos:

        - `video_check.html`: Interfaz dedicada a la clasificación de seguridad (Kids vs. Adults). Permite introducir un video y visualizar si es apto para menores.

        - `genres.html`: Interfaz diseñada para la clasificación multietiqueta, donde se muestra el género principal y las subcategorías detectadas del video.

- Carpeta `src`: Contiene todo el código correspondiente a los procesos utilizados para realizar este proyecto:

    - Carpeta `analisis`: Contiene los notebooks necesarios para el análisis exploratorio de datos (EDA) para ambos objetivos del proyecto. En ellos se documenta el estudio de distribuciones de duración, densidad de palabras (WPM) y frecuencias de categorías.

    - Carpeta `comun`: Contiene la lógica compartida y las herramientas principales del proyecto. Incluye scripts especializados para el preprocesamiento de texto y escalado de datos, utilidades de entrenamiento optimizadas para distintos algoritmos, y módulos para el filtrado, división y evaluación de modelos. Además, contiene herramientas para la generación de gráficos y la gestión del servidor.

    - Carpeta `deprecated`: Almacena versiones obsoletas y pruebas iniciales del proyecto para mantener la trazabilidad del desarrollo. Incluye las primeras implementaciones de modelos (versiones 0), scripts de testeo primarios y cuadernos con análisis preliminares de las características del dataset que han sido superados por las versiones finales en producción.

    - Carpeta `extraccion`: Implementa el pipeline de obtención de datos mediante la búsqueda de videos aleatorios y la consulta a la API de YouTube. Incluye el script principal `collect_all_data.py` para la captura de información de videos infantiles y adultos, `get_all_dfs.py` para la consolidación de múltiples conjuntos de datos en un DataFrame único, y `extraccion_y_guardado.py` para la ejecución automatizada del proceso desde la consola.

    - Carpeta `modelos`: Contiene los ficheros que entrenan la predicción de si un vídeo es apto o no para niños y la clasificación de videos por género y subcategorías.

        Para cada objetivo, se entrenaron los siguientes modelos junto con el Baseline: KNN, MLP, Naive Bayes, Random Forest, Regresión Logística, XGBoost. En Kids, además, se entrenó el algoritmo Decision Tree.

    - Carpeta `notebooks`: Contiene pruebas de experimentación y desarrollo del proyecto. Incluye ficheros iniciales de extracción de identificadores (IDs) para YouTube y YouTube Kids mediante web scraping, el diseño y prueba del pipeline completo de datos, y los experimentos preliminares de modelado con árboles de decisión para validar la capacidad predictiva de las variables seleccionadas.

    - Carpeta Private: Dentro de la carpeta src, el usuario que desee utilizar este proyecto debe crear una carpeta llamada Private. En el interior de esa carpeta, es necesario añadir un archivo llamado `claves.json` donde se almacenarán las credenciales y configuraciones necesarias para el funcionamiento del proyecto (APIs y otros servicios externos). El archivo debe tener la siguiente estructura:

        ````
        {
            "Clave_API": "Clave de la API de YouTube",
            "Url": "minio.fdi.ucm.es",
            "Access_Key": "Credenciales de acceso a MINIO",
            "Secret_Key": "Clave secreta de acceso a MINIO"
        }
        ````

> [!WARNING]
> Este archivo contiene información sensible, por lo que no debe subirse al repositorio.

### **3. 🛠️ Iniciar el entorno de desarrollo y sus dependencias**

Para desarrollar este proyecto hemos utilizado un gestor de entornos y dependencias [uv](https://github.com/astral-sh/uv), que simplifica en gran medida la configuración del entorno de desarrollo. Para poder iniciar el entorno hay que seguir una serie de pasos:

> [!IMPORTANT]
> Antes de iniciar el entorno asegurate de tener instalado uv en tu sistema. Puedes encontrar instrucciones en el reporitorio oficial de [uv](https://github.com/astral-sh/uv).

1. Nos vamos a la carpeta donde queremos guardar el repositorio.
    
    - Podemos hacerlo desde el explorador de archivos de nuestro equipo y en la dirección escribir cmd para abrir la terminal desde esa dirección.

        ![Imagen de ayuda](./Images%20readme/Direccion%20para%20replicar%20entorno.png)

    - O mediante el siguiente comando desde terminal.

        ````
        cd <dirección donde queremos guardar el repositorio>
        ````

2. Dentro de la terminal clonamos este repositorio con los siguientes comandos:

    Para clonar el repositorio:
    ````
    git clone https://github.com/UCM-GIDIA-PD1/c2526-R1.git
    ````

    Una vez clonado, nos movemos a la carpeta del repostorio:
    ````
    cd c2526-R1
    ````

3. Instalamos las dependencias del proyecto usando el siguiente comando:

    ````
    uv sync --all-groups
    ````

    Esto creará automáticamente el entorno del proyecto y descargará todas las dependecias necesarias del proyecto. (Usamos --all-goups porque tenemos distintos grupos de dependencias y, de esta manera, se descargan las dependencias del proyecto entero).

### **4. Ejecutar los scripts del proyecto**

Para ejecutar los scripts principales del proyecto, utiliza el gestor uv. Los scripts están diseñados para ser ejecutados desde la raíz del repositorio. 

Se pueden ejecutar dentro del entorno usando:

````
uv run nombre_script.py
````

Esto asegura que el script se ejecuta con al versión correcta de python y las dependencias necesarias sin necesidad de activar nada manualmente.

**4.1. Extracción y preparación de datos**

Para la generación de nuevos conjuntos de datos o la validación del pipeline de extracción, se debe ejecutar el script principal de recolección. Este permite parametrizar los datos mediante flags:

````
uv run python src/extraccion/extraccion_y_guardado.py -p 0 -i 4
````

En donde:
- `-p` (Proportion): Define la proporción de videos para adultos (0 para una muestra infantil, 1 para adultos)
- `-i` (Iterations): Determina el número de ciclos de búsqueda por palabras aleatorias, controlando el volumen final de la extracción.

**4.2. Consolidación y limpieza**

Una vez finalizadas las tandas de extracción, se deben integrar todos los ficheros `.parquet` en un único DataFrame global. Este proceso elimina los duplicados por ID y la normalización inicial:
````
uv run python src/extraccion/get_all_dfs.py  
````
**4.3. Implementación de lógica**

En este paso se define la arquitectura de los objetos de predicción en `app/train.py`. Estos objetos actúan como un "wrapper" que integra el pipeline de preprocesamiento y el modelo clasificador para automatizar el flujo completo:

     URL ➡️ Scraping de metadatos ➡️ Transformación ➡️ Clasificación.

> [!NOTE]
> Este archivo no requiere ejecución manual, ya que sus clases son instanciadas automáticamente por el servidor web.

**4.4. Ejecución de la Aplicación Web**

Para interactuar con los modelos de clasificación de forma visual y sencilla, se debe lanzar el servidor local.

````
uv run python app/main.py
````

### **5. Mejores modelos construídos**
- Para el objetivo de predicción de videos 'Made for Kids', el mejor modelo construído es Random Forest con precision de 0.94%.

- Para el objetivo de clasificación de generos, el mejor modelo construído es KNN con un F1-score de 0.69%.

### **6. Instrucciones para ejecutar la aplicación web**

Para poder ejecutar la web hemos creado un script llamado `main.py` dentro de la carpeta `app`. Para poder arrancar la web solo el necesario ejecutar el script y esperar a que la web se active.

>[!IMPORTANT]
> Para que el script funcione correctamente, en el script `app/main.py` asegurate que el `import train ...` no lleve el punto delante de train para que, al ejecutar, encuentre el archivo y no de errores.

### **7. Instrucciones para crear y ejecutar el contenedor**

Para poder ejecutar nuestra aplicación de forma aislada y portátil, usaremos los contenedores de Docker. De esta manera, la aplicación se ejecuta dentro de un contenedor que funciona igual en cualquier entorno y sistema operativo.

La configuración del contenedor está en un archivo llamado `Containerfile`, que tiene las instrucciones para descargar el entorno, las dependencias, etc.

>[!IMPORTANT]
> Para que el contenedor funcione correctamente, en el script `app/main.py` asegurate que el `import .train ...` lleve el punto delante de train para que, al ejecutar, encuentre el archivo.

La forma de ejecutar el contenedor es la siguiente:

1. Tenemos que tener instalado la herramienta Podman, que se puede instalar desde el siguiente enlace: [Podman Desktop](https://podman.io/).

    Sigue todas las instrucciones de instalación: instala las extensiones recomendadas. Con ellas podrás usar una máquina de linux para poder ejecutar los siguientes comandos. Tendrás que tenerla activa para poder usarla.

2. Una vez que tenemos todo instalado, abrimos una terminal y nos dirigimos a la dirección del proyecto para ejecutar los siguientes comandos:

    - Este comando crea la imagen con el nombre (--tag) "sfkids". Dejamos que se cree la imagen (puede tardar unos minutos)
    
        ````
        $ podman build --tag sfkids .
        ````

     - A continuación, una vez que tenemos la imagen creada, ejecutamos el contenedor, que es una instancia de la imagen que acabamos de crear (puede tardar varios minutos debido a la descarga de dependencias de la aplicación).

        ````
        $ podman run -d -p 2350:2350 --name SFKids localhost/sfkids
        ````

        El nombre que le hemos puesto a nuestro contenedor es el nombre de la aplicación para poder identificarlo (no es necesario, se le puede poner cualquier otro nombre y, en caso de no ponerlo, se genera solo automáticamente).
    
3. Para segurarnos que el contenedor esta listo, entra en la aplicación de Podman Desktop, y busca el contenedor que acabamos de crear (si le has puesto nombre, buscalo por ese nombre) y haz click sobre él. Una vez dentro verás distintas pestañas (summary, logs, Inspect, Kube, Terminal) entra en "Logs" y observa si por el final se encuentra este mensaje:

    ````
    INFO:     Started server process [67]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:2350 (Press CTRL+C to quit)
    ````

    Si ves este mensaje, significa que la aplicación se ha iniciado con éxito y ya puedes abrir el buscador y poner:

    ````
    http://localhost:2350
    ````

    Esto abrirá la aplicación web y podrás utilizarla.

    Para parar la aplicación solo tendrás que detener el contenedor usando eta instruccion:

    ````
    $ podman stop <nombre_del_contenedor>
    ````

    > [!NOTE]
    > Al ejecutarlo desde local, solo podrás acceder desde tu ordenador y no desde otro.

### **8. Equipo de desarrollo**
- Andrea Yu García Gómez
  
- Marina Gurova

- Luis López Rodríguez

- María Martín Portal
  
- Alejo Muñoz Pinilla

- Angie Ruiz Martínez
