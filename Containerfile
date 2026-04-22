# Imagen base: Python 3.13.12
FROM python:3.13.12-slim

# Copiar uv (gestor de paquetes rápido) desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la aplicación al contenedor

# 1. Copiamos primero las dependencias
COPY pyproject.toml .

# 2. Instalar las dependencias
RUN uv sync --no-cache

# 3. Copiar el resto de codigo (origen - destino)
COPY . .

# Ponemos el codigo para la aplicacion
EXPOSE 2350

# Comando para ejecutar el contenedor
CMD ["uv", "run", "python", "app/main.py"]