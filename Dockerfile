# ---- Stage 1: Build --------------------------------------------------------
# Use the official uv image with Python 3.14 to install dependencies.
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Use the lockfile for reproducible installs
ENV UV_FROZEN=1

# Install dependencies first (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Copy application source and install the project itself
COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --no-dev

# ---- Stage 2: Runtime ------------------------------------------------------
# Slim runtime image — no uv, no build tools.
FROM python:3.14-slim-trixie AS runtime

WORKDIR /app

# Copy the virtual environment from the builder (includes all deps)
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ src/
COPY scripts/ scripts/

# Copy data files — train.csv, test.csv, and the pre-built Chroma index
COPY data/ data/

# Put the venv on PATH so `uvicorn` is found directly
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
