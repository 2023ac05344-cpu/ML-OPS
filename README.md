# MLOps Pipeline - California Housing Dataset

A complete MLOps pipeline for predicting California housing prices using machine learning models.

## Architecture Overview

This project implements a full MLOps pipeline with the following components:

- **Data Versioning**: DVC for dataset tracking
- **Experiment Tracking**: MLflow for model experiments and registry
- **API Service**: FastAPI-based prediction service
- **Containerization**: Docker for deployment
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Monitoring**: Logging and metrics collection

## Project Structure

```
ML-OPS/
├── data/                   # Dataset files (tracked by DVC)
├── models/                 # Trained models
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model training
│   ├── api/               # FastAPI application
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── docker/                # Docker configuration
├── .github/               # GitHub Actions workflows
├── logs/                  # Application logs
└── notebooks/             # Jupyter notebooks for exploration
```

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd ML-OPS
   pip install -r requirements.txt
   ```

2. **Download Data**:
   ```bash
   python download_data.py
   dvc add data/california_housing.csv
   git add data/.gitignore data/california_housing.csv.dvc
   git commit -m "Add dataset"
   ```

3. **Train Models**:
   ```bash
   python src/models/train.py
   ```

4. **Run API Locally**:
   ```bash
   python src/api/main.py
   ```

5. **Docker Deployment**:
   ```bash
   docker build -t mlops-housing .
   docker run -p 8000:8000 mlops-housing
   ```

## API Endpoints

- `GET /`: Health check
- `POST /predict`: Make predictions
- `GET /metrics`: Prometheus metrics
- `GET /logs`: View recent logs

## Model Performance

The pipeline trains multiple models and selects the best performing one based on RMSE:
- Linear Regression
- Random Forest
- Gradient Boosting

## Monitoring

- Request/response logging to SQLite database
- Prometheus metrics for monitoring
- Model prediction tracking

## CI/CD Pipeline

GitHub Actions workflow that:
1. Runs tests and linting
2. Builds Docker image
3. Pushes to Docker Hub
4. Deploys to local environment