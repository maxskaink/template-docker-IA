from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import uvicorn
from loguru import logger
import sys
import os
from contextlib import asynccontextmanager

# Importar nuestro modelo
from .model import model_instance

# Configuración de logging
logger.remove()  # Remover el handler por defecto
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)
 
# Modelos Pydantic para validación
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Texto a clasificar")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('El texto no puede estar vacío')
        return v.strip()

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Lista de textos a clasificar")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('La lista de textos no puede estar vacía')
        
        cleaned_texts = []
        for text in v:
            if not isinstance(text, str):
                raise ValueError('Todos los elementos deben ser strings')
            if not text.strip():
                raise ValueError('Los textos no pueden estar vacíos')
            if len(text) > 5000:
                raise ValueError('Los textos no pueden exceder 5000 caracteres')
            cleaned_texts.append(text.strip())
        
        return cleaned_texts

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int

# Contexto de la aplicación para cargar el modelo al inicio
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando aplicación...")
    
    # Cargar modelo
    logger.info("Cargando modelo de IA...")
    success = model_instance.load_model()
    
    if not success:
        logger.error("Error crítico: No se pudo cargar el modelo")
        raise RuntimeError("No se pudo cargar el modelo de IA")
    
    logger.info("Modelo cargado exitosamente")
    logger.info("Aplicación lista para recibir requests")
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Microservicio de Clasificación de Texto con IA",
    description="API RESTful para clasificación de texto usando modelos de Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejador global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Error interno del servidor",
            "detail": "Ocurrió un error inesperado",
            "status_code": 500
        }
    )

# Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica del servicio"""
    return {
        "message": "Microservicio de Clasificación de Texto con IA",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint para verificar el estado del servicio y modelo"""
    try:
        model_info = model_instance.get_model_info()
        model_loaded = model_info.get("status") == "loaded"
        
        logger.info("Health check realizado")
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio no disponible"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Clasificar un texto individual"""
    try:
        logger.info(f"Recibida solicitud de predicción para texto: {input_data.text[:50]}...")
        
        result = model_instance.predict(input_data.text)
        
        return PredictionResponse(
            text=result["text"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
        
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la predicción"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """Clasificar múltiples textos en lote"""
    try:
        logger.info(f"Recibida solicitud de predicción en lote para {len(input_data.texts)} textos")
        
        results = model_instance.predict_batch(input_data.texts)
        
        prediction_responses = [
            PredictionResponse(
                text=result["text"],
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )
            for result in results
        ]
        
        return BatchPredictionResponse(
            results=prediction_responses,
            total_processed=len(results)
        )
        
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error en predicción en lote: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la predicción en lote"
        )

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Obtener información detallada del modelo"""
    try:
        info = model_instance.get_model_info()
        logger.info("Información del modelo solicitada")
        return info
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener información del modelo"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

