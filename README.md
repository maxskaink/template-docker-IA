# Microservicio de Clasificación de Texto con IA

## 🚀 Descripción

Esta es una plantilla de microservicio RESTful construida con **FastAPI** para servir modelos de inteligencia artificial. El servicio implementa un clasificador de texto que determina el sentimiento (positivo/negativo) de textos en español.

### Características principales:

- ⚙️ **Framework**: FastAPI con tipado estático
- 🤖 **Modelo IA**: Clasificación de texto con TF-IDF + Logistic Regression
- 🐳 **Contenerización**: Docker y Docker Compose
- 📋 **Logging**: Sistema completo con Loguru
- ✅ **Validación**: Manejo robusto de errores y validación de entradas
- 🔄 **Optimización**: Carga del modelo una sola vez al iniciar

## 📜 Estructura del Proyecto

```
plantilla_docker/
├── app/
│   ├── __init__.py
│   ├── main.py           # Punto de entrada de FastAPI
│   └── model.py          # Lógica del modelo de IA
├── requirements.txt       # Dependencias de Python
├── Dockerfile            # Configuración del contenedor
├── docker-compose.yml    # Orquestación de servicios
├── .dockerignore         # Archivos excluidos del build
└── README.md             # Esta documentación
```

## 🚀 Inicio Rápido

### Opción 1: Con Docker Compose (Recomendado)

```bash
# Clonar o descargar la plantilla
cd plantilla_docker

# Construir y ejecutar el servicio
docker-compose up --build

# El servicio estará disponible en http://localhost:8000
```

### Opción 2: Con Docker

```bash
# Construir la imagen
docker build -t ai-microservice .

# Ejecutar el contenedor
docker run -p 8000:8000 ai-microservice
```

### Opción 3: Desarrollo Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el servidor de desarrollo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 Endpoints de la API

### Documentación Interactiva
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Principales

#### 1. Health Check
```http
GET /health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "status": "loaded",
    "model_type": "TF-IDF + Logistic Regression",
    "features": 1000,
    "classes": ["negativo", "positivo"]
  }
}
```

#### 2. Predicción Individual
```http
POST /predict
Content-Type: application/json

{
  "text": "Este producto es excelente, lo recomiendo mucho"
}
```

**Respuesta:**
```json
{
  "text": "Este producto es excelente, lo recomiendo mucho",
  "prediction": "positivo",
  "confidence": 0.887,
  "probabilities": {
    "negativo": 0.113,
    "positivo": 0.887
  }
}
```

#### 3. Predicción en Lote
```http
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Me encanta este producto",
    "Muy malo, no lo recomiendo",
    "Excelente calidad"
  ]
}
```

**Respuesta:**
```json
{
  "results": [
    {
      "text": "Me encanta este producto",
      "prediction": "positivo",
      "confidence": 0.923,
      "probabilities": {"negativo": 0.077, "positivo": 0.923}
    },
    {
      "text": "Muy malo, no lo recomiendo",
      "prediction": "negativo",
      "confidence": 0.856,
      "probabilities": {"negativo": 0.856, "positivo": 0.144}
    }
  ],
  "total_processed": 3
}
```

#### 4. Información del Modelo
```http
GET /model/info
```

## 🔍 Ejemplos de Uso

### Con cURL

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Predicción individual
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Este servicio funciona perfectamente"}'

# Predicción en lote
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Excelente", "Terrible", "Regular"]}'
```

### Con Python

```python
import requests

# URL base del servicio
base_url = "http://localhost:8000"

# Predicción individual
response = requests.post(
    f"{base_url}/predict",
    json={"text": "Este producto es increíble"}
)
print(response.json())

# Predicción en lote
response = requests.post(
    f"{base_url}/predict/batch",
    json={
        "texts": [
            "Me gusta mucho",
            "No me gusta nada",
            "Está bien"
        ]
    }
)
print(response.json())
```

### Con JavaScript/Fetch

```javascript
// Predicción individual
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Este servicio es fantástico'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🔧 Personalización

### Cambiar el Modelo de IA

Para usar tu propio modelo, modifica el archivo `app/model.py`:

1. **Reemplaza la clase `TextClassificationModel`** con tu implementación
2. **Mantiene la interfaz pública**:
   - `load_model()`: Cargar modelo
   - `predict(text)`: Predicción individual
   - `predict_batch(texts)`: Predicción en lote
   - `get_model_info()`: Información del modelo

### Ejemplo con Hugging Face

```python
from transformers import pipeline

class HuggingFaceModel:
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    def load_model(self):
        try:
            self.model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        result = self.model(text)[0]
        return {
            "text": text,
            "prediction": result["label"],
            "confidence": result["score"],
            "probabilities": {result["label"]: result["score"]}
        }
```

### Variables de Entorno

Puedes configurar el servicio con variables de entorno:

```bash
# Puerto del servicio
export PORT=8000

# Nivel de logging
export LOG_LEVEL=INFO

# Ruta del modelo
export MODEL_PATH=/app/models/custom_model.joblib
```

## 📋 Logs y Monitoreo

### Estructura de Logs

- **Consola**: Logs informativos con formato simple
- **Archivo**: Logs detallados en `logs/app.log` con rotación automática

### Ver Logs del Contenedor

```bash
# Logs en tiempo real
docker-compose logs -f ai-microservice

# Últimas 100 líneas
docker-compose logs --tail=100 ai-microservice
```

### Métricas de Health Check

El endpoint `/health` proporciona:
- Estado del servicio
- Estado de carga del modelo
- Información del modelo actual
- Métricas de rendimiento

## 🚪 Despliegue en Producción

### Consideraciones de Seguridad

1. **CORS**: Modifica `allow_origins` en `main.py` para especificar dominios permitidos
2. **HTTPS**: Usa un proxy reverso (Nginx, Traefik) para SSL/TLS
3. **Secrets**: No hardcodees claves API o secretos

### Optimizaciones

1. **Multi-stage build**: Optimiza el Dockerfile para imágenes más pequeñas
2. **Resource limits**: Configura límites de CPU/memoria en docker-compose.yml
3. **Caching**: Implementa caching para predicciones frecuentes

### Ejemplo con Nginx

```nginx
server {
    listen 80;
    server_name tu-dominio.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Escalabilidad

### Horizontal Scaling

```yaml
# docker-compose.yml para múltiples instancias
services:
  ai-microservice:
    # ... configuración existente ...
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ai-microservice
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-microservice
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-microservice
  template:
    metadata:
      labels:
        app: ai-microservice
    spec:
      containers:
      - name: ai-microservice
        image: ai-microservice:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## 🛠️ Troubleshooting

### Problemas Comunes

1. **Modelo no carga**:
   ```bash
   # Verificar logs
   docker-compose logs ai-microservice | grep -i error
   
   # Verificar salud del servicio
   curl http://localhost:8000/health
   ```

2. **Puerto ocupado**:
   ```bash
   # Cambiar puerto en docker-compose.yml
   ports:
     - "8001:8000"  # Puerto 8001 en lugar de 8000
   ```

3. **Memoria insuficiente**:
   ```bash
   # Aumentar memoria disponible para Docker
   # O usar un modelo más pequeño
   ```

### Debugging

```bash
# Entrar al contenedor
docker-compose exec ai-microservice bash

# Verificar estructura de archivos
ls -la /app/

# Verificar logs en tiempo real
tail -f logs/app.log
```

## 📚 Recursos Adicionales

- [Documentación de FastAPI](https://fastapi.tiangolo.com/)
- [Guía de Docker](https://docs.docker.com/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Transformers de Hugging Face](https://huggingface.co/transformers/)

## 🤝 Contribución
¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Envía un Pull Request

## 📜 Licencia

Este proyecto es de código abierto bajo la Licencia MIT.

---

¿🚀 ¿Listo para servir tu modelo de IA? ¡Clona esta plantilla y personaliza según tus necesidades!

