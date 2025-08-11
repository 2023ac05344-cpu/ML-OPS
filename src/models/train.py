import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training class with MLflow integration."""

    def __init__(self, data_path: str = "data/california_housing.csv"):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor()
        self.best_model = None
        self.best_score = float("inf")

    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        df = self.preprocessor.load_data(self.data_path)
        return self.preprocessor.preprocess_data(df)

    def train_models(self):
        """Train multiple models and track experiments with MLflow."""
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()

        # Define models to train
        models = {
            "linear_regression": {"model": LinearRegression(), "params": {}},
            "random_forest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                },
            },
            "gradient_boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
            },
        }

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(
            "sqlite:////app/mlflow.db"
            if os.environ.get("MLFLOW_IN_DOCKER") == "1"
            else "sqlite:///mlflow.db"
        )

        # Train each model
        for model_name, model_config in models.items():
            logger.info(f"Training {model_name}")

            with mlflow.start_run(
                run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                # Set model parameters
                model = model_config["model"]
                if model_config["params"]:
                    model.set_params(**model_config["params"])

                # Log parameters
                mlflow.log_params(model_config["params"])

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Infer model signature and provide input example to remove warning
                signature = infer_signature(X_test, y_pred)
                input_example = X_test.iloc[:1]

                # Log model with signature and input example
                mlflow.sklearn.log_model(
                    model,
                    f"{model_name}_model",
                    signature=signature,
                    input_example=input_example,
                )

                logger.info(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

                # Track best model
                if rmse < self.best_score:
                    self.best_score = rmse
                    self.best_model = model
                    self.best_model_name = model_name

                    # Save best model locally
                    os.makedirs("models", exist_ok=True)
                    import shutil

                    if os.path.exists("models/best_model"):
                        shutil.rmtree("models/best_model")
                    mlflow.sklearn.save_model(model, f"models/best_model")

                    # Register best model in MLflow
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/{model_name}_model",
                        "housing_price_predictor",
                    )

        logger.info(
            f"Best model: {self.best_model_name} with RMSE: {self.best_score:.4f}"
        )
        return self.best_model, self.preprocessor

    def save_preprocessor(self, preprocessor, path: str = "models/preprocessor.pkl"):
        """Save the preprocessor for later use."""
        import pickle

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to {path}")


def main():
    """Main training function."""
    trainer = ModelTrainer()
    best_model, preprocessor = trainer.train_models()
    trainer.save_preprocessor(preprocessor)

    # Save best model locally and register in MLflow
    import shutil

    os.makedirs("models", exist_ok=True)
    preprocessor_path = os.path.join("models", "preprocessor.pkl")
    # Save preprocessor
    pd.to_pickle(preprocessor, preprocessor_path)
    logger.info("Preprocessor saved to models/preprocessor.pkl")

    # Save best model via MLflow as the production candidate
    with mlflow.start_run(run_name="register_best_model"):
        signature = infer_signature(
            trainer.preprocessor.preprocess_data(
                trainer.preprocessor.load_data(trainer.data_path)
            )[0].iloc[:1],
            best_model.predict(
                trainer.preprocessor.preprocess_data(
                    trainer.preprocessor.load_data(trainer.data_path)
                )[0].iloc[:1]
            ),
        )
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            signature=signature,
            input_example=trainer.preprocessor.preprocess_data(
                trainer.preprocessor.load_data(trainer.data_path)
            )[0].iloc[:1],
            registered_model_name="housing_price_predictor",
        )
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
