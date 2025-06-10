# Usar imagen base oficial de Python
FROM python:3.11-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Crear usuario no-root para mayor seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app/ ./app/

# Crear directorios necesarios y establecer permisos
RUN mkdir -p logs app/models && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 775 /app/logs && \
    chmod -R 775 /app/app/models

# Cambiar al usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Comando de salud para Docker
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

