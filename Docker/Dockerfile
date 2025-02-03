# Stage 1: Build dependencies in a temporary container
FROM python:3.9-slim AS builder

WORKDIR /app

# Install dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Minimal runtime container
FROM python:3.9-alpine

WORKDIR /app

# Ensure Python is installed in Alpine
RUN apk add --no-cache python3 py3-pip libstdc++ bash

# Set Python3 as the default command (fixing missing `/usr/local/bin/python3`)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY run.py .
#COPY model_checkpoint.pth .
COPY data/ ./data/

# Expose Flask API port
EXPOSE 5000

# Use the correct path for Python3
CMD ["/usr/bin/python3", "run.py"]

