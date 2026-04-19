FROM python:3.11-slim

WORKDIR /mlflow

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@db:5432/mlflow", \
     "--default-artifact-root", "/mlruns"]
