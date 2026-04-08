import polars as pl
import numpy as np
import shap
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

class FeatureEngineer:
    def __init__(self,use_diff=False,use_pca=False,n_components=10):
        self.base_cols = None
        self.first_diff_cols = None
        self.second_diff_cols = None
        self.history =[]

        # ★ ここが重要
        self.target_cols = ["含水率", "含水率_log"]

        self.scaler = StandardScaler()
        self.use_diff = use_diff
        self.use_pca = use_pca
        self.pca = PCA(n_components=n_components) if use_pca else None

    def fit(self, df):
        self.base_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Int64)
            and c not in self.target_cols
        ]

        if self.use_diff:
            df_0 = self.add_diff(df)
            df = self.add_diff2(df_0) 

        # base_cols更新（重要）
        #self.base_cols = [
        #    c for c in df.columns
        #    if c not in self.target_cols
        #    and df[c].dtype in (pl.Float64, pl.Int64)
        #]
        self.base_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
        


        X = df.select(self.base_cols).to_numpy().astype("float32")
        X = self.scaler.fit_transform(X)

        if self.use_pca:
            self.pca.fit(X)

        return self
    
    def add_diff(self, df: pl.DataFrame):
        new_cols = [...]
        cols = self.base_cols.copy()

        # ===== 次微分（1次diffから作る）=====
        diff_exprs = [
            (pl.col(cols[i+1]) - pl.col(cols[i])).alias(f"{cols[i+1]}_diff")
            for i in range(len(cols) - 1)
        ]

        df = df.with_columns(diff_exprs)

        # ===== 2次微分（1次diffから作る）=====
        diff_cols = [f"{cols[i+1]}_diff" for i in range(len(cols) - 1)]

        diff_diff_exprs = [
            (pl.col(diff_cols[i+1]) - pl.col(diff_cols[i])).alias(f"{diff_cols[i+1]}_diff2")
            for i in range(len(diff_cols) - 1)
        ]

        df = df.with_columns(diff_diff_exprs)
        self.diff_cols = [f"{cols[i+1]}_diff" for i in range(len(cols) - 1)]

        self.history.append({
            "type": "diff1",
            "input_cols": cols,
            "output_cols": self.diff_cols,
        })

        return df
    
    def add_diff2(self, df: pl.DataFrame):
        new_cols = [...]
        if not hasattr(self, "diff_cols"):
            raise ValueError("先にadd_diffを実行してください")

        diff_cols = self.diff_cols

        diff2_exprs = [
            (pl.col(diff_cols[i+1]) - pl.col(diff_cols[i])).alias(f"{diff_cols[i+1]}_diff2")
            for i in range(len(diff_cols) - 1)
        ]

        df = df.with_columns(diff2_exprs)

        # ★ 追加：2次微分列を記録
        self.diff2_cols = [f"{diff_cols[i+1]}_diff2" for i in range(len(diff_cols) - 1)]

        self.history.append({
            "type": "diff2",
            "input_cols": self.diff_cols,
            "output_cols": self.diff2_cols,
        })

        return df


    def transform(self, df: pl.DataFrame):
        if self.base_cols is None:
            raise ValueError("fitが先に必要")
        
        if self.use_diff:
            df = self.add_diff(df)
            df = self.add_diff2(df)

        # 列チェック
        missing_cols = [c for c in self.base_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"不足列: {missing_cols}")

        X = df.select(self.base_cols).to_numpy().astype("float32")
        X = self.scaler.transform(X)

        if self.use_pca:
            X_pca = self.pca.transform(X)
            pca_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]
            df_pca = pl.DataFrame(X_pca, schema=pca_cols)
            return df.with_columns(df_pca)
        

        return df
    

    def show_shap(self, df: pl.DataFrame, model, max_display=20):
        import shap
        import pandas as pd
        import numpy as np

        # ===== transform（fit済み前提）=====
        df_transformed = self.transform(df)

        # ===== 数値列だけ取得 =====
        X = df_transformed.select(self.base_cols).to_numpy().astype("float32")

        # ===== SHAP =====
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # ===== DataFrame化 =====
        shap_df = pd.DataFrame(shap_values, columns=self.base_cols)

        # ===== 可視化（summary）=====
        shap.summary_plot(shap_values, X, feature_names=self.base_cols, max_display=max_display)

        # ===== 平均寄与度 =====
        importance = shap_df.abs().mean().sort_values(ascending=False)

        print("\n=== SHAP importance ===")
        print(importance.head(max_display))

        return importance




class MoisturePipeline:
    def __init__(self,params=None,use_diff=False,use_pca=False):

        self.params = params or {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 5,
            "n_jobs": -1,
            "verbosity": -1 
        }

        self.fe = FeatureEngineer(use_pca=use_pca)

        self.target_cols = ["含水率"]

        self.drop_cols = ["樹種", "sample number"] + self.target_cols

        self.model = None
        self.feature_cols = None  
    

    def fit(self, train_df: pl.DataFrame):
        # ===== 特徴量生成 =====
        self.fe.fit(train_df)
        train_df = self.fe.transform(train_df)
        # ===== 特徴量を確定 =====
        cols_to_drop = [c for c in self.drop_cols if c in train_df.columns]+["species number"]

        self.feature_cols = [
            c for c in train_df.columns if c not in cols_to_drop
        ]

                # ===== X =====
        X = train_df.select(self.feature_cols).to_numpy().astype("float32")

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


import mlflow.pyfunc

class FullPipelineModel(mlflow.pyfunc.PythonModel):

    def __init__(self, pipe):
        self.pipe = pipe

    def predict(self, context, model_input):
        return self.pipe.predict(model_input)