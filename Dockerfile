FROM python:3.10

WORKDIR /linumpy/

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools and wheel
RUN pip install --upgrade pip setuptools wheel build

# Install with verbose output
COPY . .
RUN pip install --no-cache-dir -v -e .