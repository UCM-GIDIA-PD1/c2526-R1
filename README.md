# **SAFE FOR KIDS (SFKids)**

![Imagen portada](./Images%20readme/imagen%20portada.png)

## **Índice**

1. [Descripción del proyecto](#1-descripción-proyecto)

2. [Estructura del repositorio](#2-estructura-del-repositorio)

3. [Instalación del entorno de desarrollo y sus dependecias](#3-️-iniciar-el-entorno-de-desarrollo-y-sus-dependencias)

4. [Descarga de la base de datos](#4-descarga-de-la-base-de-datos)

### **1. Descripción proyecto**

En este proyecto principal de este proyecto es desarrollar una herramienta capar de determinar de manera autónoma si un video va a ser apto o no para un nivel de madurez determinado. El objetivo es facilitar al usuario un filtrado selectivo de la información que considere inadecuada.

En paralelo, este sistema busca  asignar una categoría princpal y diversas subcategorías, proporcionando una organización más detallada y precisa.

### **2. Estructura del repositorio**

**Archivos:**

- `.gitignore`: contiene los archivos que no se deben subir al git desde el repositorio local.

- `.python-version`: contiene la version de python usada en el proyecto.

- `pyproyect.toml` y `uv.lock`: contiene la configuración del entorno, con las dependencias y las versiones correspondientes.

**Carpetas:**

- Carpeta `src`: contiene todo el código. En esta carpeta encontramos procesos utilizados para realizar este proyecto.

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

    Este clona el repositorio.
    ````
    git clone https://github.com/UCM-GIDIA-PD1/c2526-R1.git
    ````

    Una vez clonado, nos movemos a la carpeta del repostorio.
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

### **4. Descarga de la base de datos**

