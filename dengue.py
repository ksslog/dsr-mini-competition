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

# Michele:
from preprocessing import preprocessing

# load the datasets

features = pd.read_csv('./data/dengue_features_train.csv')
labels = pd.read_csv('./data/dengue_labels_train.csv')
test_features = pd.read_csv('./data/dengue_features_test.csv')

# Preprocessing:
# Drop columns not used
# Encode the city column
# Not fill missing values (will be done in the pipeline)
# ....
X, test_features = preprocessing(features, test_features)

# Prepare y:
y = labels.loc[:,'total_cases']

# Split sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

# Pasquale:

# Instantiate pipeline and fit model
pipeline = make_pipeline(
    SimpleImputer(),
    RandomForestRegressor()
)
pipeline.fit(X_train, y_train)

# Sergii:

# Obtain predictions and score

pred = pipeline.predict(X_test)
score = mean_absolute_error(y_test, pred)
print('MAE', score)

# Plot results
# ...

# Write out csv with prediction
