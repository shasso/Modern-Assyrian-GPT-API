# Use NVIDIA PyTorch container matching NVIDIA playbook instructions
# Documentation page lists 25.09-py3; keeping newer 25.10-py3 is fine, but you can pin 25.09 if needed.
FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install additional Python dependencies
# PyTorch is already included in the base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy model files (these will be mounted or copied at runtime)
# COPY models/ ./models/

# Copy the application code
COPY main.py .

# Expose port
EXPOSE 8000

# NVIDIA / CUDA environment hints for container runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# Copy entrypoint script to emit CUDA diagnostics before starting
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application via entrypoint (prints diagnostics then execs)
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "main.py"]