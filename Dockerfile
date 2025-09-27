FROM python:3.10-slim

# Install system dependencies (needed for faiss, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements and install as root
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Switch to non-root
USER user

# Copy app code
COPY --chown=user . .

# Expose port (Hugging Face Spaces default)
EXPOSE 7860

# Environment variable (set properly in Space secrets instead of here)
ENV GROQ_API_KEY=""

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]