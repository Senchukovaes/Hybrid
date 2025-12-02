# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# install build deps for some packages (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
WORKDIR /app/src

# make script executable
RUN chmod +x hybrid_stego.py

ENTRYPOINT ["python", "hybrid_stego.py"]
