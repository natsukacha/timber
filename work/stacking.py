# stacking.py

import mlflow
import mlflow.pyfunc
import polars as pl
from ml_pipeline import MoisturePipeline


from sklearn.model_selection import KFold
import numpy as np

from lightgbm import LGBMRegressor


class FullPipelineModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, context, model_input):
        return self.pipeline.predict(model_input)


def train_species_models(train_df):

    species_list = train_df["species number"].unique().to_list()
    models = {}

    for sp in species_list:

        sp_train = train_df.filter(pl.col("species number") == sp)

        if sp_train.height == 0:
            continue

        pipe = MoisturePipeline(species=f"sp_{sp}")
        rmse = pipe.fit(sp_train)

        mlflow.log_param("species", sp)
        mlflow.log_metric("rmse", rmse)

        mlflow.pyfunc.log_model(
            artifact_path=f"model_sp_{sp}",
            python_model=FullPipelineModel(pipe)
        )

        models[sp] = pipe

    return models, species_list


def create_oof(train_df, species_list):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    meta_features = np.zeros((train_df.height, len(species_list)))
    y = train_df["含水率"].to_numpy()

    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(train_df.height))):

        train_fold = train_df[tr_idx]
        val_fold = train_df[val_idx]

        fold_models = {}

        for i, sp in enumerate(species_list):

            sp_train = train_fold.filter(pl.col("species number") == sp)

            if sp_train.height == 0:
                continue

            pipe = MoisturePipeline(species=f"sp_{sp}")
            pipe.fit(sp_train)

            fold_models[sp] = pipe

        for i, sp in enumerate(species_list):

            if sp not in fold_models:
                continue

            sp_val = val_fold.filter(pl.col("species number") == sp)

            if sp_val.height == 0:
                continue

            pred = fold_models[sp].predict(sp_val)

            meta_features[val_idx, i] = pred

    return meta_features, y


def train_meta_model(X_meta, y):
    model = LGBMRegressor()
    model.fit(X_meta, y)
    return model

class StackingModel(mlflow.pyfunc.PythonModel):

    def __init__(self, models, meta_model, species_list):
        self.models = models
        self.meta_model = meta_model
        self.species_list = species_list

    def predict(self, context, df):

        meta_features = []

        for sp in self.species_list:

            if sp not in self.models:
                meta_features.append(np.zeros(len(df)))
                continue

            sp_df = df.filter(pl.col("species number") == sp)

            if sp_df.height == 0:
                meta_features.append(np.zeros(len(df)))
                continue

            pred = self.models[sp].predict(sp_df)

            full_pred = np.zeros(len(df))
            full_pred[:len(pred)] = pred

            meta_features.append(full_pred)

        X_meta = np.column_stack(meta_features)

        return self.meta_model.predict(X_meta)