from typing import List
import pandas as pd
import numpy as np


def downcast_floats(df: pd.DataFrame) -> None:
    df.loc[:, df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)


def cast_booleans(df: pd.DataFrame, features: List[str]) -> None:
    df.loc[:, features] = df.loc[:, features].fillna(False).astype(bool)
