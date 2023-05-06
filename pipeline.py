from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import numpy as np
import pandas as pd


pipeline = make_pipeline(
    KNNImputer(n_neighbors=3),
    # SimpleImputer(),
    StandardScaler(),
    # LinearRegression(), # just for laughs
    # GradientBoostingRegressor(n_estimators=3000),
    ExtraTreesRegressor(n_estimators=5000),
    # RandomForestRegressor(ccp_alpha=0, n_estimators=1000),
    # XGBRegressor(n_estimators=5000),
)


train_features = pd.read_csv("./data/dengue_features_train.csv")
test_features = pd.read_csv("./data/dengue_features_test.csv")

train_labels = pd.read_csv("./data/dengue_labels_train.csv")

X = train_features.drop(
    columns=["week_start_date", "city", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]
)
y = train_labels.loc[:, "total_cases"]

### The simplest and most supid test possible:
pipeline.fit(X, y)
pipeline.score(X, y)
cross_val_score(pipeline, X, y, cv=5)

### let's split in X_train and X_val

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)

pipeline.fit(X_train, y_train)
pipeline.score(X_val, y_val)
cross_val_score(pipeline, X_train, y_train, cv=5)
cross_val_score(pipeline, X_val, y_val, cv=5)

### let's add features: five weeks' memory


# X_test = test_features.drop(columns=["week_start_date", "city"])
X_extra = pd.concat(
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
X_train, X_val, y_train, y_val = train_test_split(
    X_extra, y, random_state=2, test_size=0.20
)

pipeline.fit(X_train, y_train)
pipeline.score(X_val, y_val)


cross_val_score(pipeline, X_train, y_train, cv=5).mean()
cross_val_score(pipeline, X_val, y_val, cv=5).mean()
