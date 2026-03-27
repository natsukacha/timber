FROM python:3.11-slim

# システムパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
RUN python -m pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    tqdm \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    lightgbm \
    xgboost \
    optuna \
    polars \
    pyarrow \
    mlflow

# 作業ディレクトリ
WORKDIR /work

# ポート
EXPOSE 8888

# Jupyter起動（tokenなし）
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token="]