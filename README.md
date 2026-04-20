# **SAFE FOR KIDS (SFKids)**

![Imagen portada](./Images%20readme/imagen%20portada.png)

## **Índice**

1. [Descripción de los objetivos](#1-descripción-de-los-objetivos)

2. [Estructura del repositorio](#2-estructura-del-repositorio)

3. [Iniciar el entorno de desarrollo y sus dependecias](#3-️-iniciar-el-entorno-de-desarrollo-y-sus-dependencias)

4. [Descarga de la base de datos](#4-descarga-de-la-base-de-datos)
5. [Mejores modelos construídos](#5-mejores-modelos-construídos)

8. [Equipo de desarrollo](#8-equipo-de-desarrollo)

### **1. Descripción de los objetivos**

1. Desarrollar una herramienta capaz de determinar de manera autónoma si un video de Youtube va a estar destinado o no para (niños). Para ello se usa un modelo de aprendizaje autónomo entrenado para que sea capaz de decidir, a partir de un umbral de decisión, si es apto o no.
2. Asignar un género principal y múltiples subcategorías a cada video para identificar tanto la temática general (e.g., Educación, Deportes, Videojuegos) como detalles específicos, permitiendo a cualquier usuario filtrar contenido de manera precisa según sus intereses o necesidades pedagógicas.

### **2. Estructura del repositorio**

**Archivos:**

- `.gitignore`: Contiene los archivos que no se deben subir al git desde el repositorio local.

- `.python-version`: Contiene la version de python usada en el proyecto.

- `pyproyect.toml` y `uv.lock`: Contiene la configuración del entorno, con las dependencias y las versiones correspondientes.

**Carpetas:**
- Carpeta `app`: Contiene los ficheros necesarios para ejecutar la aplicación web (API y Frontend), organizada de la siguiente manera:
     - `main.py`: Punto de entrada de la aplicación. Contiene la lógica del servidor, la definición de los endpoints de la API REST y la carga de los modelos entrenados para generar las predicciones en tiempo real.
     - `train.py`: Script destinado al re-entrenamiento o ajuste final de los modelos antes de su puesta en producción en el entorno web.
     - Carpeta `templates`: Almacena las interfaces de usuario (HTML) que permiten la interacción con los modelos:
          - video_check.html: Interfaz dedicada a la clasificación de seguridad (Kids vs. Adults). Permite introducir un video y visualizar si es apto para menores.
          - genres.html: Interfaz diseñada para la clasificación multietiqueta, donde se muestra el género principal y las subcategorías detectadas del video.
   
- Carpeta `src`: Contiene todo el código. En esta carpeta encontramos todos los procesos utilizados para realizar este proyecto:

    - Extracción y transformación (carpeta `extraccion`):
        - `extraccion_y_guardado.py`: Archivo que se puede llamar desde la consola para iniciar la extracción de datos

        - `collect_all_data.py`: Archivo principal que se encarga de la extracción de datos de videos aleatorios, utilizando los IDs y la API oficial de Youtube. Se extraen videos tanto para adultos como para videos.

        - `get_all_dfs.py`: Archivo que une todos los dataframes de los videos extraidos.
  - Carpeta `Modelos`: 
  - Carpeta Private: _Importante_
        - Dentro de la carpeta src, el usuario que desee utilizar este proyecto debe crear una carpeta llamada "*Private*".
            En el interior de esa carpeta, es necesario añadir un archivo llamado `claves.json` donde se almacenarán las credenciales y configuraciones necesarias para el funcionamiento del proyecto (APIs y otros servicios externos).
            
            El archivo debe tener la siguiente estructura:
                
                {
                    "Clave_API": "Clave de la API de YouTube",
                    "Url": "minio.fdi.ucm.es",
                    "Access_Key": "Credenciales de acceso a MINIO",
                    "Secret_Key": "Clave secreta de acceso a MINIO"
                }

            *Nota importante:* _Este archivo contiene información sensible, por lo que no debe subirse al repositorio._

- Carpeta `Images readme`: contiene las imágenes de este archivo.


### **3. 🛠️ Iniciar el entorno de desarrollo y sus dependencias**

Para desarrollar este proyecto hemos utilizado un gestor de entornos y dependencias [uv](https://github.com/astral-sh/uv), que simplifica en gran medida la configuración del entorno de desarrollo.
Para poder iniciar el entorno hay que seguir una serie de pasos:

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
    uv sync
    ````

    Esto creará automáticamente el entorno del proyecto y descargará todas las dependecias necesarias del proyecto.

4. Se pueden ejecutar scripts dentro del entorno usando:
    ````
    uv run nombre_script.py
    ````

    Esto asegura que el script se ejecuta con al versión correcta de python y las dependencias necesarias sin necesidad de activar nada manualmente.

### **4. Descarga de datos**
Para descargar nuevos videos de YouTube y YouTube Kids para la base de datos, se debe ejecutar el archivo parametrizacion.py, llamándolo como 
````
uv run python src/parametrizacion.py.
````
Por defecto, al ejecutar el código de este archivo se guardarán 500 videos nuevos, de los cuales aproximadamente 80% están destinados a adultos y 20% a niños.
Se pueden proporcionar parámetros para el número de videos a guardar, la fecha a partir de la cuál se van a guardar los videos y la proporción de videos para adultos.
Por ejemplo: 
````
uv run python src/parametrizacion.py -n 1000 -p 0.5.   
````

### **Instrucciones para ejecutar los scripts del proyecto**
### **5. Mejores modelos construídos**
- Para el objetivo de predicción de videos 'Made for Kids', el mejor modelo construído es Random Forest con precision de 0.94%.
- Para el objetivo de predicción de generos, el mejor modelo construído es KNN con un F1-score de 0.69%.

### **6. Instrucciones para ejecutar la aplicación web
Aquí va la explicación para acceder a la aplicación, si hay un servidor o se deber ejecutar local, etc.

### **7. Instrucciones para crear y ejecutar el contenedor**
Aquí va la explicación de Docker

### **8. Equipo de desarrollo**
- Andrea Yu García Gómez
  
- Marina Gurova

- Luis López Rodríguez

- María Martín Portal
  
- Alejo Muñoz Pinilla

- Angie Ruiz Martínez
