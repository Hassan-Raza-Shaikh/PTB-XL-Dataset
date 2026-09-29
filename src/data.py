import ast
import os

import numpy as np
import pandas as pd
import yaml

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_metadata(config):
    """Reads ptbxl_database.csv and adds one binary column per diagnostic superclass."""
    df = pd.read_csv(config["data"]["database_csv"], index_col="ecg_id")
    scp_df = pd.read_csv(config["data"]["scp_csv"], index_col=0)

    diag = scp_df[scp_df.diagnostic == 1]
    code_to_superclass = {code: cls for code, cls in zip(diag.index, diag.diagnostic_class) if pd.notna(cls)}

    codes = df.scp_codes.apply(ast.literal_eval)
    superclasses = codes.apply(lambda d: {code_to_superclass[c] for c in d if c in code_to_superclass})
    for sc in SUPERCLASSES:
        df[sc] = superclasses.apply(lambda s: int(sc in s))
    return df


def split_masks(df, config):
    folds = df.strat_fold
    return (
        folds.isin(config["split"]["train_folds"]).values,
        folds.isin(config["split"]["val_folds"]).values,
        folds.isin(config["split"]["test_folds"]).values,
    )


def processed_path(config, name):
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    return os.path.join(config["data"]["processed_dir"], name)


def label_matrix(df):
    return df[SUPERCLASSES].values.astype(np.float32)
