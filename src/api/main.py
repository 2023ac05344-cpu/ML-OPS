import os
import sys
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
import json

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import mlflow.sklearn
import pickle
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter("predictions_total", "Total number of predictions")
PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds", "Time spent processing prediction"
)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="California Housing Price Predictor",
    description="MLOps pipeline for predicting California housing prices",
    version="1.0.0",
)


# Pydantic models for request/response
class HousingInput(BaseModel):
    """Input model for housing prediction."""

    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms")
    AveBedrms: float = Field(..., description="Average number of bedrooms")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    prediction: float
    model_version: str
    timestamp: str


class LogEntry(BaseModel):
    """Model for log entries."""

    timestamp: str
    input_data: Dict[str, float]
    prediction: float
    model_version: str


class DatabaseManager:
    """Database manager for logging predictions."""

    def __init__(self, db_path: str = "logs/predictions.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction REAL NOT NULL,
                model_version TEXT NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def log_prediction(
        self, input_data: Dict[str, float], prediction: float, model_version: str
    ):
        """Log a prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (timestamp, input_data, prediction, model_version)
            VALUES (?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                json.dumps(input_data),
                prediction,
                model_version,
            ),
        )

        conn.commit()
        conn.close()

    def get_recent_logs(self, limit: int = 10) -> List[LogEntry]:
        """Get recent prediction logs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, input_data, prediction, model_version
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        logs = []
        for row in rows:
            logs.append(
                LogEntry(
                    timestamp=row[0],
                    input_data=json.loads(row[1]),
                    prediction=row[2],
                    model_version=row[3],
                )
            )

        return logs


# Global variables
db_manager = DatabaseManager()
model = None
preprocessor = None
model_version = "unknown"


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_version

    try:
        # Load model from MLflow
        model_path = "models/best_model"
        if os.path.exists(model_path):
            model = mlflow.sklearn.load_model(model_path)
            model_version = "local_best_model"
        else:
            # Try to load from MLflow registry
            model = mlflow.sklearn.load_model("models:/housing_price_predictor/latest")
            model_version = "mlflow_registry_latest"

        # Load preprocessor
        with open("models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        logger.info(f"Model loaded successfully. Version: {model_version}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting California Housing Price Predictor API")
    load_model()


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "California Housing Price Predictor"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HousingInput):
    """Make a prediction for housing price."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Start timing
        start_time = datetime.now()

        # Convert input to dict
        input_dict = input_data.dict()

        # Transform input
        input_transformed = preprocessor.transform_input(input_dict)

        # Make prediction
        prediction = model.predict(input_transformed)[0]

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Update metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_DURATION.observe(duration)

        # Log prediction
        db_manager.log_prediction(input_dict, prediction, model_version)

        logger.info(f"Prediction made: {prediction:.2f} in {duration:.3f}s")

        return PredictionResponse(
            prediction=float(prediction),
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/logs", response_model=List[LogEntry])
async def get_logs(limit: int = 10):
    """Get recent prediction logs."""
    return db_manager.get_recent_logs(limit)


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_version": model_version,
        "model_type": type(model).__name__,
        "feature_columns": preprocessor.get_feature_columns() if preprocessor else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
