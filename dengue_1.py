import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from post_processing import post_processing

# Michele:
from preprocessing import preprocessing


# load the datasets

features = pd.read_csv('./data/dengue_features_train.csv')
labels = pd.read_csv('./data/dengue_labels_train.csv')
test_features = pd.read_csv('./data/dengue_features_test.csv')

# Preprocessing:
# Drop columns not used
# Encode the city column
# (Nan values will be fixed in the pipeline)
# ....
X, test_features = preprocessing(features, test_features)

# Prepare y:
y = labels.loc[:,'total_cases']

# Split sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Pasquale:

# Instantiate pipeline and fit model
pipeline = make_pipeline(
    KNNImputer(n_neighbors=3),
    StandardScaler(),
    ExtraTreesRegressor(n_estimators=500),
)

pipeline.fit(X_train, y_train)
pipeline.score(X_train, y_train)
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Sergii:

# Obtain predictions and score
pred = pipeline.predict(X_test)
trained = pipeline.predict(X_train)
score = mean_absolute_error(y_test, pred)
print('MAE', score)

png_file_name = 'save_as_a_png.png'
csv_file_name = 'out.csv'
model_name    = 'ExtraTreesRegressor'
post_processing(X_train, y_train, trained, X_test,
                y_test,pred, png_file_name, csv_file_name, model_name)
