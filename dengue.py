import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocessing
from post_processing import post_processing

# load the datasets

features = pd.read_csv('./data/dengue_features_train.csv')
labels = pd.read_csv('./data/dengue_labels_train.csv')
test_features = pd.read_csv('./data/dengue_features_test.csv')

# Preprocessing:
# Drop columns not used
# Encode the city column
# (Nan values will be fixed in the pipeline)
# ....
X, test_features_clean = preprocessing(features, test_features)

# Separate data
dfX = []
dfX.append(X[X.city == 'sj'])
dfX.append(X[X.city == 'iq'])

df_test_features_clean = []
df_test_features = []
df_test_features_clean.append(test_features_clean[test_features_clean.city == 'sj'])
df_test_features_clean.append(test_features_clean[test_features_clean.city == 'iq'])
df_test_features.append(test_features[test_features.city == 'sj'])
df_test_features.append(test_features[test_features.city == 'iq'])


df_labels = []
df_labels.append(labels[labels.city == 'sj'])
df_labels.append(labels[labels.city == 'iq'])

df21 = []

for i in range(2):

    # Prepare X and y:
    y = df_labels[i].loc[:,'total_cases']
    dfX[i].drop('city', axis=1, inplace=True)

    # Split sets:
    X_train, X_val, y_train, y_val = train_test_split(dfX[i], y, test_size=0.2, shuffle=False)

    # Instantiate pipeline and fit model
    pipeline = make_pipeline(
        KNNImputer(n_neighbors=1),
        StandardScaler(),
        ensemble.ExtraTreesRegressor(n_estimators=500)
        )
    pipeline.fit(X_train, y_train)

    # Validate and score

    pred_val = pipeline.predict(X_val)
    score = mean_absolute_error(y_val, pred_val)
    print(i, 'MAE on validation set:', score)

    pred_train = pipeline.predict(X_train)
    png_file_name = str(i) + 'save_as_a_png.png'
    post_processing(X_train, y_train, pred_train, X_val, y_val, pred_val, png_file_name)

    # Final training of the model (without validation set)
    pipeline.fit(dfX[i], y)

    # Drop 'city'
    df_test_features_clean[i].drop('city', axis=1, inplace=True)
    
    # Obtain predictions and score

    pred_to_submit = pipeline.predict(df_test_features_clean[i])
    df2 = df_test_features[i][['city', 'year', 'weekofyear']]
    df2['total_cases'] = pred_to_submit.round().astype(int)
    df21.append(df2)


df_out = pd.concat([df21[0], df21[1]], axis=0)
df_out.to_csv('out.csv', index=False)
