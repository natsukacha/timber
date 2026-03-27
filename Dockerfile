FROM python:3.11-slim

# システムパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
RUN pip install --no-cache-dir \
    jupyterlab \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    lightgbm \
    xgboost \
    optuna \
    polars \
    pyarrow

# 作業ディレクトリ
WORKDIR /work

# ポート
EXPOSE 8888

# JupyterLab起動
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888","--no-browser", "--allow-root", "--ServerApp.root_dir=/work"]
