# **SAFE FOR KIDS (SFKids)**

## **Índice**

1. [Descripción de los objetivos](#1-descripción-de-los-objetivos)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Iniciar el entorno de desarrollo y sus dependecias](#3-iniciar-el-entorno-de-desarrollo-y-sus-dependencias)

### 1. Descripción de los objetivos
El objetivo principal de este proyecto desarrollar un sistema de aprendizaje automático.

### 2. Estructura del repositorio
- La carpeta **src** contiene todo el código. En esta carpeta encontramos procesos utilizados para realizar este proyecto.

### 3. Iniciar el entorno de desarrollo y sus dependencias
Para desarrollar este proyecto hemos utilizado un gestor de entornos y dependencias [uv](https://github.com/astral-sh/uv), que simplifica en gran medida la configuración del entorno de desarrollo.
Para poder iniciar el entorno hay que seguir una serie de pasos:

[!IMPORTANT]

**IMPORTANTE**: Antes de 

1. Nos vamos a la carpeta donde queremos guardar el repositorio.

- Podemos hacerlo desde el explorador de archivos y en la dirección escribir cmd para abrir la terminal desde esa dirección.

- O mediante el siguiente comando desde terminal.

````
cd <dirección donde queremos guardar el repositorio>
````

2. Clonamos este repositorio:

````
git clone https://github.com/UCM-GIDIA-PD1/c2526-R1.git
cd c2526-R1
````

3. Instalamos las dependencias del proyecto usando el siguiente comando:

````
uv sync
````

Esto creará automáticamente el entorno del proyecto y descargará todas las dependecias necesarias en nuestro 
