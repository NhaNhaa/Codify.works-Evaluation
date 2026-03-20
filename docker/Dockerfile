FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install torch CPU-only first to avoid heavy CUDA packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required runtime directories
RUN mkdir -p data/inputs data/outputs data/chroma_storage logs

EXPOSE 8000

CMD ["python", "-m", "backend.main"]