FROM python:3.11-slim

# システムパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt .

# Pythonパッケージ
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token="]