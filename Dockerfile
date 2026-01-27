FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.27" "pytest>=8.0" "httpx>=0.27" "ruff>=0.4"

# Copy source
COPY app /app/app
COPY tests /app/tests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]