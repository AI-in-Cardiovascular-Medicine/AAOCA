FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends make libpq-dev git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
# Set the working directory
WORKDIR /app
COPY ./web_service_models /app/web_service_models
COPY script.py utils.py preprocessing_cropping.py  /app

ENTRYPOINT ["python", "script.py"]