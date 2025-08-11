import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    def setup_method(self):
        """Set up test data."""
        self.preprocessor = DataPreprocessor()

        # Create sample data
        self.sample_data = pd.DataFrame(
            {
                "MedInc": [8.3252, 8.3014, 7.2574],
                "HouseAge": [41.0, 21.0, 52.0],
                "AveRooms": [6.984127, 6.238137, 8.288136],
                "AveBedrms": [1.023810, 0.971880, 1.073446],
                "Population": [322.0, 2401.0, 496.0],
                "AveOccup": [2.555556, 2.109842, 2.802260],
                "Latitude": [37.88, 37.86, 37.85],
                "Longitude": [-122.23, -122.22, -122.24],
                "median_house_value": [452600.0, 358500.0, 352100.0],
            }
        )

    def test_load_data(self, tmp_path):
        """Test data loading functionality."""
        # Create a temporary CSV file
        csv_path = tmp_path / "test_data.csv"
        self.sample_data.to_csv(csv_path, index=False)

        # Test loading
        loaded_data = self.preprocessor.load_data(str(csv_path))
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == self.sample_data.shape
        assert list(loaded_data.columns) == list(self.sample_data.columns)

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(
            self.sample_data
        )

        # Check that data is split correctly
        assert len(X_train) + len(X_test) == len(self.sample_data)
        assert len(y_train) + len(y_test) == len(self.sample_data)

        # Check that features are scaled
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)

        # Check that feature columns are set
        assert self.preprocessor.feature_columns is not None
        assert len(self.preprocessor.feature_columns) == 8  # All features except target

    def test_transform_input(self):
        """Test input transformation for prediction."""
        # First preprocess some data to set up the scaler
        self.preprocessor.preprocess_data(self.sample_data)

        # Test input transformation
        input_data = {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23,
        }

        transformed = self.preprocessor.transform_input(input_data)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (1, 8)  # One sample, 8 features

    def test_get_feature_columns(self):
        """Test getting feature columns."""
        # Initially should be None
        assert self.preprocessor.get_feature_columns() is None

        # After preprocessing, should return feature columns
        self.preprocessor.preprocess_data(self.sample_data)
        feature_cols = self.preprocessor.get_feature_columns()
        assert isinstance(feature_cols, list)
        assert len(feature_cols) == 8
        assert "median_house_value" not in feature_cols
