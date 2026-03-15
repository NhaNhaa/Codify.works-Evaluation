FROM python:3.11-slim

WORKDIR /app

# Install torch CPU-only FIRST (skips 2GB+ of NVIDIA CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create required directories
RUN mkdir -p data/inputs/students data/outputs data/chroma_storage logs

EXPOSE 8000

CMD ["python", "-m", "backend.main"]