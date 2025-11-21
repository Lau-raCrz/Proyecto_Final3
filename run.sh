#!/bin/bash

echo "🚀 Iniciando aplicación Streamlit..."

xhost +local:docker

docker compose up --build

xhost -local:docker

echo "✅ Aplicación detenida"
