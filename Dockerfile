# Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# Instalar dependencias del sistema operativo para Visión Artificial (Tesseract y Poppler)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar requirements e instalar librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de tu código fuente y la base de datos (db_itesco)
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para encender Jaguarundi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]