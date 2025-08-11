import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing class for California Housing dataset."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the California Housing dataset."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    def preprocess_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ):
        """Preprocess the data for training."""
        logger.info("Starting data preprocessing")

        # Define feature columns (all except target)
        self.feature_columns = [col for col in df.columns if col != "MedHouseVal"]

        # Split features and target
        X = df[self.feature_columns]
        y = df["MedHouseVal"]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)

        logger.info(
            f"Preprocessing complete. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}"
        )

        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform_input(self, input_data: dict) -> np.ndarray:
        """Transform input data for prediction."""
        # Convert input dict to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[self.feature_columns]

        # Scale the input
        input_scaled = self.scaler.transform(input_df)

        return input_scaled

    def get_feature_columns(self) -> list:
        """Get the feature column names."""
        return self.feature_columns
