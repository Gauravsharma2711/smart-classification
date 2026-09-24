# Production Dockerfile for Smart Waste Classifier (FR-20)
# Designed for Hugging Face Spaces, Render, AWS, and Cloud Run

FROM python:3.12-slim

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /workspace

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./

# Install python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application and source code
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY reports/ ./reports/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860" \
    PORT=7860

# Expose standard Gradio port
EXPOSE 7860

# Run Gradio application
CMD ["uv", "run", "python", "-m", "waste_classifier.cli", "app", "--host", "0.0.0.0", "--port", "7860"]
