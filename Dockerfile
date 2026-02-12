# 1. Use a small, official Python image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# 4. Set working directory inside the container
WORKDIR /app

# 5. Install system dependencies (minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy dependency definitions first (better caching)
COPY pyproject.toml ./

# 7. Install Python dependencies
RUN pip install --upgrade pip \
    && pip install .

# 8. Copy application code
COPY app ./app
COPY prompts ./prompts
COPY frontend ./frontend

# 9. Expose the API port
EXPOSE 8000

# 10. Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
