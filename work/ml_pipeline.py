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
    def __init__(self,use_diff=False,use_conv=False,use_band=False,
                 use_pca=False,n_components=10):
        self.base_cols = None
        self.first_diff_cols = []
        self.second_diff_cols = []
        self.band_feature=[]
        self.band_centers=[]
        self.one_demention_conv_cols = []
        self.feature_cols=[]
        self.history =[]

        self.target_cols = ["含水率", "含水率_log"]
        self.id_etc = ["sample number","species number","樹種"]

        self.scaler = StandardScaler()
        self.use_diff = use_diff
        self.use_conv = use_conv
        self.use_pca  = use_pca
        self.use_band = use_band
        self.pca = PCA(n_components=n_components) if use_pca else None

    def fit(self, df):
        non_features = self.target_cols + self.id_etc
        self.original_base_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Int64)
            and c not in non_features
        ]



        if self.use_diff:
            self.first_diff_cols = [
                f"{self.original_base_cols[i+1]}_diff"
                for i in range(len(self.original_base_cols) - 1)
            ]
            df = self._apply_diff(df)

        if self.use_conv:
            df = self.one_demention_conv(df)

        # 例：SHAPから決めた中心波長
        self.band_centers = [7500, 4700, 10000]

        self.col_to_wavelength = {
           c: float(c) for c in self.original_base_cols
        }

        df = self._apply_band_feature(df)




        self.feature_cols = (
            self.original_base_cols
            + self.first_diff_cols
            + self.one_demention_conv_cols
            + self.band_feature
        )

        X = df.select(self.feature_cols).to_numpy().astype("float32")
        X = self.scaler.fit_transform(X)

        if self.use_pca:
            self.pca.fit(X)

        return self

    def _apply_diff(self, df: pl.DataFrame):
        # 1次微分
        diff_cols = [
            f"{self.original_base_cols[i+1]}_diff"
            for i in range(len(self.original_base_cols) - 1)
        ]
            
        df = df.with_columns([
            (pl.col(self.original_base_cols[i+1]) - pl.col(self.original_base_cols[i]))
            .alias(self.first_diff_cols[i])
            for i in range(len(self.original_base_cols) - 1)
        ])

        self.first_diff_cols = diff_cols


        # 2次微分
        #df = df.with_columns([
        #    (pl.col(self.first_diff_cols[i+1]) - pl.col(self.first_diff_cols[i]))
        #    .alias(self.second_diff_cols[i])
        #    for i in range(len(self.first_diff_cols) - 1)
        #])

        return df
    
    def _apply_band_feature(self, df: pl.DataFrame, width=50):
        band_cols = []

        for center in self.band_centers:
            # 対象波長選択
            selected_cols = [
                c for c in self.original_base_cols
                if abs(self.col_to_wavelength[c] - center) <= width
            ]

            if len(selected_cols) == 0:
                continue

            col_name = f"band_{int(center)}"

            df = df.with_columns(
                (pl.sum_horizontal([pl.col(c) for c in selected_cols]) / len(selected_cols)
                ).alias(col_name)
            )

            band_cols.append(col_name)

        self.band_feature = band_cols
        return df


    def one_demention_conv(self, df: pl.DataFrame):
        import torch
        import torch.nn as nn
        import numpy as np
        import polars as pl

        # ===== 元特徴量取得 =====
        X = df.select(self.original_base_cols).to_numpy().astype("float32")

        # (N, L) → (N, 1, L)
        X = np.expand_dims(X, axis=1)
        X_tensor = torch.tensor(X)

        # ===== 固定Conv定義（smoothing + 微分っぽい）=====
        conv = nn.Conv1d(
            in_channels=1,
            out_channels=4,   # フィルタ数
            kernel_size=5,
            padding=2,
            bias=False
        )

        # ===== カーネルを手動設定（ここがキモ）=====
        with torch.no_grad():
            kernels = np.array([
                [1, 2, 3, 2, 1],     # smoothing
                [-1, -2, 0, 2, 1],   # 1次微分風
                [1, -2, 0, -2, 1],   # 2次微分風
                [-1, 0, 2, 0, -1],   # エッジ強調
            ], dtype=np.float32)

            kernels = kernels[:, None, :]  # (out, in, k)
            conv.weight.copy_(torch.tensor(kernels))

        conv.eval()

        # ===== 畳み込み =====
        with torch.no_grad():
            out = conv(X_tensor)  # (N, 4, L)

        pool = nn.AdaptiveAvgPool1d(50)
        out = pool(out).numpy()


        # ===== flattenして特徴量化 =====
        N, C, L = out.shape
        out = out.reshape(N, C * L)

        # ===== カラム名 =====
        conv_cols = [
            f"conv_{c}_{i}"
            for c in range(C)
            for i in range(L)
        ]

        self.one_demention_conv_cols = [
        f"conv_{c}_{i}"
        for c in range(C)
        for i in range(L)
        ]

        conv_df = pl.DataFrame(out, schema=conv_cols)

        # ===== 結合 =====
        df = pl.concat([df, conv_df], how="horizontal")

        return df



    def transform(self, df: pl.DataFrame):
        if self.feature_cols is None:
            raise ValueError("fitが先に必要")
        
        if self.use_diff:
            #diffの2重生成を防ぐ
            df = self._apply_diff(df)

        if self.use_conv:
            df = self.one_demention_conv(df)

        if self.use_band:
            df = self._apply_band_feature(df)

        # 列チェック
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"不足列: {missing_cols}")

        X = df.select(self.feature_cols).to_numpy().astype("float32")
        X = self.scaler.transform(X)

        #if self.use_pca:
        #    X_pca = self.pca.transform(X)
        #    pca_cols = [f"pca_{i}" for i in range(X_pca.shape[1])]
        #    df_pca = pl.DataFrame(X_pca, schema=pca_cols)
        #    return df.with_columns(df_pca)
        

        return df
    

    def show_shap(self, df: pl.DataFrame, model,feature_cols, max_display=20):
        import shap
        import pandas as pd
        import numpy as np

        # ===== transform（fit済み前提）=====
        df_transformed = self.transform(df)

        # ===== 数値列だけ取得 =====
        X = df_transformed.select(feature_cols).to_numpy().astype("float32")

        # ===== SHAP =====
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # ===== DataFrame化 =====
        shap_df = pd.DataFrame(shap_values, columns=feature_cols)

        # ===== 可視化（summary）=====
        shap.summary_plot(shap_values, X, feature_names=feature_cols)

        # ===== 平均寄与度 =====
        importance = shap_df.abs().mean().sort_values(ascending=False)

        print("\n=== SHAP importance ===")
        print(importance.head(max_display))

        # ===== DataFrameとして整形 =====
        importance_df = (
            importance
            .reset_index()
            .rename(columns={"index": "feature", 0: "importance"})
            .sort_values("importance", ascending=False)
        )

        return importance_df




class MoisturePipeline:
    def __init__(self,params=None,use_diff=False,use_pca=False,
    use_conv=False,use_band=False):
        self.use_diff = use_diff
        self.use_conv = use_conv
        self.use_band = use_band  
        self.use_pca = use_pca

        self.params = params or {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 5,
            "n_jobs": -1,
            "verbosity": -1 
        }

        self.fe = FeatureEngineer(use_diff=use_diff,
            use_conv=use_conv,use_band=use_band,use_pca=use_pca)

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

        self.feature_cols = self.fe.feature_cols

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