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

# Install Python dependencies from requirements.txt with verbose output
COPY requirements.txt ./
RUN pip install --no-cache-dir -v -r requirements.txt

COPY . .

RUN pip install -e .