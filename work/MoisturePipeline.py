

from FeatureEngineer import FeatureEngineer

import polars as pl
import numpy as np
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold



class MoisturePipeline:
    def __init__(self,params=None,use_diff=False,use_pca=False,
    use_conv=False,use_band=False,use_sg=False):
        self.use_diff = use_diff
        self.use_conv = use_conv
        self.use_band = use_band  
        self.use_pca = use_pca
        self.use_sg = use_sg


        self.params = params or {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 5,
            "n_jobs": -1,
            "verbosity": -1 
        }

        self.fe = FeatureEngineer(use_diff=use_diff,
            use_conv=use_conv,use_band=use_band,use_sg=use_sg,use_pca=use_pca)

        self.target_cols = ["含水率"]

        self.drop_cols = ["樹種", "sample number"] + self.target_cols

        self.model = None
        self.feature_cols = None  
    


    def fit(self, train_df: pl.DataFrame):

        # ===== 特徴量生成 =====
        self.fe.fit(train_df)
        train_df = self.fe.transform(train_df)

        self.feature_cols = self.fe.feature_cols

        # ===== X =====
        X = train_df.select(self.feature_cols).to_numpy().astype("float32")

        # ===== y（log変換）=====
        y_raw = train_df["含水率"].to_numpy()
        y = np.log1p(y_raw)

        # ===== KFold =====
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        rmses = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # ===== 学習 =====
            model = LGBMRegressor(**self.params)
            model.fit(X_train, y_train)

            # ===== 予測 =====
            pred_log = model.predict(X_valid)

            # ===== 元スケール評価 =====
            pred = np.expm1(pred_log)
            y_valid_raw = np.expm1(y_valid)

            rmse = np.sqrt(mean_squared_error(y_valid_raw, pred))
            rmses.append(rmse)

            print(f"fold {fold}: RMSE={rmse:.4f}")

        # ===== CV平均 =====
        mean_rmse = np.mean(rmses)
        print(f"CV RMSE: {mean_rmse:.4f}")

        # ===== 最終モデル（全データで再学習）=====
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X, y)

        return mean_rmse

    def preprocess(self, df: pl.DataFrame):
        # ===== 必須チェック =====
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"testに不足している列: {missing_cols}")

        return df.select(self.feature_cols).to_numpy().astype("float32")

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

        with mlflow.start_run():

            # ===== 対数変換をかける ====
            # パラメータ
            mlflow.log_params(self.params)

            rmse = self.fit(train_df)

            mlflow.log_metric("rmse", rmse)

            mlflow.lightgbm.log_model(self.model, "model")

            print(f"RMSE:", rmse)
