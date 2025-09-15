FROM python:3.12

WORKDIR /linumpy/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libhdf5-dev \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools and wheel
RUN pip install --upgrade pip setuptools wheel build

# Install with verbose output
COPY linumpy ./linumpy
COPY scripts ./scripts
COPY pyproject.toml requirements.txt README.md setup.py ./
RUN pip install --no-cache-dir -v -e .
