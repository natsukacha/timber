import polars as pl
import numpy as np
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class MoisturePipeline:
    def __init__(self, species: str, params=None):
        """
        species: 樹種（例: gingo）
        """
        self.species = species  # ← 追加（重要）

        self.params = params or {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 5,
            "n_jobs": -1
        }

        self.drop_cols = ["含水率", "樹種", "sample number", "species number"]

        # 学習後に入るもの
        self.model = None

    def preprocess(self, df: pl.DataFrame, is_train=True):
        """
        前処理ブロック
        """
        X = df.drop(self.drop_cols).to_numpy()

        if is_train:
            y = df["含水率_log"].to_numpy()
            return X, y
        else:
            return X

    def fit(self, train_df: pl.DataFrame):
        """
        trainデータのブロック
        """
        X, y = self.preprocess(train_df, is_train=True)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))

        return rmse

    def predict(self, test_df: pl.DataFrame):
        """
        testデータのブロック
        """
        X = self.preprocess(test_df, is_train=False)
        return self.model.predict(X)

    def run_mlflow(self, train_df: pl.DataFrame):
        """
        実行管理（MLflow）
        """
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(self.species)  # ← 種別ごとに分ける

        with mlflow.start_run():

            # ===== ここが実務で効く =====
            mlflow.log_param("species", self.species)

            # パラメータ
            mlflow.log_params(self.params)

            rmse = self.fit(train_df)

            mlflow.log_metric("rmse", rmse)

            mlflow.lightgbm.log_model(self.model, "model")

            print(f"[{self.species}] RMSE:", rmse)