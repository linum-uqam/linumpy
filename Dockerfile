FROM python:3.14

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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

# Install with uv
COPY linumpy ./linumpy
COPY scripts ./scripts
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev
