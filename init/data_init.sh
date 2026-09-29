#!/bin/bash
echo ">> Checking if dataset is present..."

if [[ ! -f "/dataset/flights.csv" ]]; then
    mkdir -p dataset

    curl -fL -o /tmp/flight-delays.zip \
      https://www.kaggle.com/api/v1/datasets/download/usdot/flight-delays

    unzip -o /tmp/flight-delays.zip -d /dataset

    rm dataset/airlines.csv
    rm dataset/airports.csv

fi

echo ">> Dataset should be ready to go"
