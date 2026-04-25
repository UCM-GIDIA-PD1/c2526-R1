# Imagen base: Python 3.13 en su versión mínima (sin herramientas innecesarias)
FROM python:3.13-slim

# Copiar uv (gestor de paquetes rápido) desde su imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# --- Optimización de caché de Docker ---
# Docker construye la imagen en capas. Si un fichero no cambia, reutiliza la capa cacheada.
# Por eso instalamos las dependencias ANTES de copiar el código:
# así, si solo cambia el código, Docker no reinstala las dependencias.

# 1. Copiar solo el fichero de dependencias
COPY pyproject.toml .

# 2. Instalar dependencias (esta capa se cachea mientras pyproject.toml no cambie)
RUN uv sync --no-cache

# 3. Copiar el resto del código (modelos, plantillas, estáticos, etc.)
COPY . .

# Puerto en el que escucha la aplicación (documentativo, no abre el puerto por sí solo)
EXPOSE 8000

# Comando que se ejecuta al arrancar el contenedor
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000"]

# -----------------------------------------------------------------------------
# Construir la imagen:
#   docker build -t webapp .
#
# Ejecutar un contenedor:
#   docker run -p 8000:8000 webapp
#
# La opción -p 8000:8000 mapea el puerto 8000 del contenedor al 8000 del host.
# Después se puede acceder a la app en http://localhost:8000
# -----------------------------------------------------------------------------