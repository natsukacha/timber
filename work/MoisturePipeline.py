
import utils
import FeaturEngineer

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
