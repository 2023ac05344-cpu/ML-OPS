#!/bin/bash

# MLOps Pipeline Deployment Script

set -e

echo "🚀 Starting MLOps Pipeline Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Step 1: Download and prepare data
print_status "Step 1: Downloading and preparing data..."
python download_data.py

# Step 2: Train models
print_status "Step 2: Training models..."
python src/models/train.py

# Step 3: Build Docker image
print_status "Step 3: Building Docker image..."
docker build -t mlops-housing .

# Step 4: Run the application
print_status "Step 4: Starting the application..."
docker run -d \
    --name mlops-housing-api \
    -p 8000:8000 \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    mlops-housing

# Step 5: Wait for application to start
print_status "Step 5: Waiting for application to start..."
sleep 10

# Step 6: Health check
print_status "Step 6: Performing health check..."
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    print_status "✅ Application is healthy and running!"
    print_status "🌐 API is available at: http://localhost:8000"
    print_status "📊 API Documentation: http://localhost:8000/docs"
    print_status "📈 Metrics: http://localhost:8000/metrics"
    print_status "📝 Logs: http://localhost:8000/logs"
else
    print_error "❌ Health check failed. Application may not be running properly."
    exit 1
fi

# Step 7: Optional - Start monitoring stack
if [ "$1" = "--with-monitoring" ]; then
    print_status "Step 7: Starting monitoring stack..."
    docker-compose up -d prometheus grafana
    
    print_status "📊 Prometheus is available at: http://localhost:9090"
    print_status "📈 Grafana is available at: http://localhost:3000 (admin/admin)"
fi

print_status "🎉 Deployment completed successfully!"

# Show running containers
echo ""
print_status "Running containers:"
docker ps --filter "name=mlops-housing" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 