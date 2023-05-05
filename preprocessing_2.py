import pandas as pd
import numpy as np


def preprocessing(X):
    # Drop 'year, 'week_start_date' and duplicate column 'precipitation_amt_mm'
    X = X.drop(
        columns=[
            "week_start_date",
            "city",
            "ndvi_ne",
            "ndvi_nw",
            "ndvi_se",
            "ndvi_sw",
        ],
    )

    # Adding to each observation the data of the past five weeks
    A = pd.concat(
        [
            X,
            X.shift(periods=1),
            X.shift(periods=2),
            X.shift(periods=3),
            X.shift(periods=4),
            X.shift(periods=5),
        ],
        axis=1,
    )

    return X
