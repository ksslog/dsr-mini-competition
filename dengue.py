import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# load the datasets

features = pd.read_csv('./data/dengue_features_train.csv')
labels = pd.read_csv('./data/dengue_labels_train.csv')

# Drop columns not used

train_features.drop('week_start_date', axis=1, inplace=True)

# Encode the city column

# ...

# Fill missing values

train_features.fillna(method='ffill', inplace=True)

# Instantiate pipeline and fit model
pipeline = make_pipeline(
    SimpleImputer(),
    RandomForestRegressor()
)
pipeline.fit(X_train, y_train)

# Obtain predictions and score

pred = pipeline.predict(X_test)
score = mean_absolute_error(y_test, pred)

# Plot results
# ...

# Write out csv with prediction
