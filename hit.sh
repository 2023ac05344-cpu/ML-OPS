#!/bin/bash

# Define the API endpoint
API_URL="http://localhost:8080/predict"

# Define the input data in JSON format
INPUT_DATA='{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}'

# Run 10 predictions
for i in {1..10}; do
  curl -X POST "$API_URL" \
       -H "Content-Type: application/json" \
       -d "$INPUT_DATA"
done
