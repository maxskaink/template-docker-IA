import os
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class TextClassificationModel:
    """
    Modelo de clasificación de texto usando TF-IDF + Logistic Regression
    """
    
    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.is_loaded = False
        self.model_path = "/tmp/text_classifier.joblib"
        
    def load_model(self) -> bool:
        """
        Carga el modelo preentrenado desde disco
        """
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info(f"Modelo cargado exitosamente desde {self.model_path}")
                return True
            else:
                logger.warning(f"Archivo de modelo no encontrado en {self.model_path}. Creando modelo de ejemplo...")
                self._create_example_model()
                return True
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            return False
    
    def _create_example_model(self):
        """
        Crea un modelo de ejemplo para demostración
        """
        try:
            # Datos de ejemplo para entrenamiento
            texts = [
                "Este producto es excelente, muy recomendado",
                "No me gustó para nada, muy malo",
                "Producto increíble, superó mis expectativas",
                "Terrible calidad, no lo compren",
                "Buena relación calidad-precio",
                "Decepcionante, esperaba más",
                "Fantástico servicio y producto",
                "Pésima experiencia de compra"
            ]
            
            labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positivo, 0: negativo
            
            # Crear pipeline con TF-IDF + Logistic Regression
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ))
            ])
            
            # Entrenar el modelo
            self.model.fit(texts, labels)
            
            # Guardar el modelo
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            self.is_loaded = True
            logger.info("Modelo de ejemplo creado y guardado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al crear modelo de ejemplo: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Realiza predicción sobre un texto
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Modelo no cargado. Llama a load_model() primero.")
        
        try:
            # Predicción
            prediction = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]
            
            # Mapear etiquetas
            label_map = {0: "negativo", 1: "positivo"}
            predicted_label = label_map[prediction]
            
            # Confidence score
            confidence = float(max(probabilities))
            
            result = {
                "text": text,
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": {
                    "negativo": float(probabilities[0]),
                    "positivo": float(probabilities[1])
                }
            }
            
            logger.info(f"Predicción realizada: {predicted_label} (confianza: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Realiza predicciones en lote
        """
        if not self.is_loaded or self.model is None:
            raise ValueError("Modelo no cargado. Llama a load_model() primero.")
        
        try:
            predictions = self.model.predict(texts)
            probabilities = self.model.predict_proba(texts)
            
            label_map = {0: "negativo", 1: "positivo"}
            
            results = []
            for i, text in enumerate(texts):
                predicted_label = label_map[predictions[i]]
                confidence = float(max(probabilities[i]))
                
                result = {
                    "text": text,
                    "prediction": predicted_label,
                    "confidence": confidence,
                    "probabilities": {
                        "negativo": float(probabilities[i][0]),
                        "positivo": float(probabilities[i][1])
                    }
                }
                results.append(result)
            
            logger.info(f"Predicciones en lote realizadas para {len(texts)} textos")
            return results
            
        except Exception as e:
            logger.error(f"Error en predicción en lote: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo
        """
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": "TF-IDF + Logistic Regression",
            "features": self.model.named_steps['tfidf'].get_feature_names_out().shape[0] if hasattr(self.model.named_steps['tfidf'], 'get_feature_names_out') else "unknown",
            "classes": ["negativo", "positivo"]
        }

# Instancia global del modelo (singleton)
model_instance = TextClassificationModel()

