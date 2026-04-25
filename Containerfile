# Imagen base: Python 3.13.12
FROM python:3.13.12-slim

# Copiar uv (gestor de paquetes rápido) desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Necesitamos unas dependecias de C para compilar algunas librerias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la aplicación al contenedor

# 1. Copiamos primero las dependencias
COPY pyproject.toml uv.lock README.md .

# 2. Instalar las dependencias
RUN uv sync --no-cache

# 3. Copiar el resto de codigo (origen - destino)
COPY . .

# Ponemos el codigo para la aplicacion
EXPOSE 2350

# Comando para ejecutar el contenedor
# Añadimos el directorio actual al PYTHONPATH para que Python encuentre la carpeta 'app'
ENV PYTHONPATH=/app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2350"]