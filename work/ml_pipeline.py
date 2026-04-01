import polars as pl
import numpy as np
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self, n_components=10):
        self.base_cols = None

        # ★ ここが重要
        self.target_cols = ["含水率", "含水率_log"]

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    def fit(self, df: pl.DataFrame):
        # ===== target完全除外 =====
        self.base_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Int64)
            and c not in self.target_cols
        ]

        X = df.select(self.base_cols).to_numpy()
        X = self.scaler.fit_transform(X)

        self.pca.fit(X)

        return self

    def transform(self, df: pl.DataFrame):
        if self.base_cols is None:
            raise ValueError("fitが先に必要")

        # ===== 列チェック =====
        missing_cols = [c for c in self.base_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"不足列: {missing_cols}")

        X = df.select(self.base_cols).to_numpy()
        X = self.scaler.transform(X)
        X_pca = self.pca.transform(X)

        pca_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]
        df_pca = pl.DataFrame(X_pca, schema=pca_cols)

        return df.with_columns(df_pca)
    




class MoisturePipeline:
    def __init__(self, species: str, params=None):
        self.species = species
        

        self.params = params or {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 5,
            "n_jobs": -1,
            "verbosity": -1 
        }

        self.target_cols = ["含水率", "含水率_log"]

        self.drop_cols = ["樹種", "sample number", 
                          "species number"] + self.target_cols

        self.model = None
        self.feature_cols = None  # ← これが最重要
        self.fe = FeatureEngineer()#calss feature engより追加

    def fit(self, train_df: pl.DataFrame):
        # ===== 特徴量生成 =====
        self.fe.fit(train_df)
        train_df = self.fe.transform(train_df)
        # ===== 特徴量を確定 =====
        cols_to_drop = [c for c in self.drop_cols if c in train_df.columns]

        self.feature_cols = [
            c for c in train_df.columns if c not in cols_to_drop
        ]

                # ===== X =====
        X = train_df.select(self.feature_cols).to_numpy()

        # ===== y（log変換）=====
        y_raw = train_df["含水率"].to_numpy()
        y = np.log1p(y_raw)  # ← ここが変更点

        # ===== 分割 =====
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ===== 学習 =====
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)

        # ===== 評価（log空間で）=====
        pred_log = self.model.predict(X_valid)

        # 元スケールに戻して評価（重要）
        pred = np.expm1(pred_log)
        y_valid_raw = np.expm1(y_valid)

        rmse = np.sqrt(mean_squared_error(y_valid_raw, pred))

        return rmse

    def preprocess(self, df: pl.DataFrame):
        # ===== 必須チェック =====
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"testに不足している列: {missing_cols}")

        return df.select(self.feature_cols).to_numpy()

    def predict(self, test_df: pl.DataFrame):
        # ===== testにtargetが混入していたらエラー =====
        if "含水率" in test_df.columns:
            raise ValueError("testデータに目的変数が含まれています")

        test_df = self.fe.transform(test_df)
        X = self.preprocess(test_df)

        # ===== 予測（log空間）=====
        pred_log = self.model.predict(X)

        # ===== 元スケールに戻す =====
        pred = np.expm1(pred_log)

        return pred

    def run_mlflow(self, train_df: pl.DataFrame):
        """
        実行管理（MLflow）
        """
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment(self.species)  # ← 種別ごとに分ける

        with mlflow.start_run():

            # ===== 対数変換をかける =====
            mlflow.log_param("species", self.species)

            # パラメータ
            mlflow.log_params(self.params)

            rmse = self.fit(train_df)

            mlflow.log_metric("rmse", rmse)

            mlflow.lightgbm.log_model(self.model, "model")

            print(f"[{self.species}] RMSE:", rmse)