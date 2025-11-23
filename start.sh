#!/bin/bash

# Syriac GPT API Startup Script

echo "🚀 Starting Syriac GPT API Service"
echo "=================================="

# Check if Docker is available
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo "✅ Docker and Docker Compose found"

    # Build and start the service
    echo "🏗️  Building and starting Docker containers..."
    docker compose up --build -d

    # Wait for service to be healthy
    echo "⏳ Waiting for service to be ready..."
    sleep 10

    # Check if service is running
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo "✅ Service is running!"
        echo ""
        echo "🌐 API Documentation: http://localhost:8000/docs"
        echo "🏥 Health Check: http://localhost:8000/health"
        echo ""
        echo "🧪 Run tests with: python test_api.py"
        echo "📊 View logs with: docker compose logs -f"
        echo "🛑 Stop service with: docker compose down"
    else
        echo "❌ Service failed to start. Check logs:"
        docker compose logs
    fi

else
    echo "❌ Docker or Docker Compose not found"
    echo "Please install Docker and Docker Compose first"
    exit 1
fi