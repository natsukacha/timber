FROM python:3.11-slim

RUN python -m pip install --no-cache-dir \
    mlflow \
    psycopg2-binary

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server","--host", "0.0.0.0","--port", "5000","--backend-store-uri", "postgresql://mlflow:mlflow@db:5432/mlflow","--default-artifact-root", "/mlruns","--allowed-hosts", "mlflow,mlflow:5000,localhost,127.0.0.1,localhost:5000,127.0.0.1:5000"]