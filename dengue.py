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

from sklearn.model_selection import cross_val_score

from pipelines import pipeline_extra_trees
from pipelines import pipeline_random_forest
from pipelines import pipeline_xgb
from pipelines import pipeline_scaler

from sklearn.model_selection import GridSearchCV

# Michele:
from preprocessing_2 import preprocessing

# load the datasets

features = pd.read_csv("./data/dengue_features_train.csv")
labels = pd.read_csv("./data/dengue_labels_train.csv")
test_features = pd.read_csv("./data/dengue_features_test.csv")

### To get an idea, we do a rough visualization right at the start:

# we drop two non numerical columns
features_drop = features.drop(columns=["city", "week_start_date"])

#we rescale and fill some gaps

pipeline_scaler.fit(features_drop)
features_scaled = pd.DataFrame(pipeline_scaler.fit_transform(features_drop), columns=features_drop.columns)

# Prepare y:
y = labels.loc[:, "total_cases"]

# we rescale the training  y label values
y_scaled = (y - y.mean())/y.var()**0.5

# we plot all columns against y_scaled

T = features_scaled

for i in T.columns:
    fig, ax = plt.subplots()
    sns.lineplot(data=T, x=T.index, y=y_scaled, label="Cases")
    sns.lineplot(data=T, x=T.index, y=T.loc[:,i], label= i)
    ax.legend()
    fig.show()



### Preprocessing:
# Drop columns not used
# Encode the city column
# (Nan values will be fixed in the pipeline)
# ....
X, test_features = preprocessing(features), preprocessing(test_features)
#X, test_features = preprocessing(features, test_features)

# Prepare y:
y = labels.loc[:, "total_cases"]

# Split sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Pasquale:

# Perform a grid search with, for example, ExtraTrees
# Apparently, the best score is obtained by n_estimators = 500

parameters = {
    "extratreesregressor__n_estimators": [100,300,500, 800, 1000,1500, 2000, 3000],
    "extratreesregressor__max_depth": [None],
    # "extratreesregressor__criterion": ["absolute_error"],
    "extratreesregressor__max_features": [None],
}

clf = GridSearchCV(pipeline_extra_trees, parameters)

clf.fit(X_train, y_train)

print("best parameters: " + str(clf.best_params_) + "\n")
print("best score: " + str(clf.best_score_) + "\n")
print(clf.cv_results_)

# Fit and Evaluate on the the validation ("_test") set

pipeline_extra_trees.set_params(extratreesregressor__n_estimators= 1000)
pipeline_extra_trees.fit(X_train, y_train)
pipeline_extra_trees.score(X_test, y_test)
cross_val_score(pipeline_extra_trees, X_train, y_train, cv=5)
cross_val_score(pipeline_extra_trees, X_test, y_test, cv=5)

pipeline_xgb.set_params(xgbregressor__n_estimators= 5000)
pipeline_xgb.fit(X_train, y_train)
pipeline_xgb.score(X_test, y_test)

# Sergii:

# Obtain predictions and score

pred = pipeline_extra_trees.predict(X_test)
score = mean_absolute_error(y_test, pred)
print("MAE", score)

# Plot results
#

import seaborn as sns
pred = pipeline_extra_trees.predict(X_test)
fig, ax = plt.subplots()
sns.lineplot(data=X_test, x=X_test.index, y=y_test, label="test")
sns.lineplot(data=X_test, x=X_test.index, y=pred, label="predict")
ax.legend()
fig.show()
fig.
# Write out csv with prediction


yg = l
xg = T.loc[:, 'reanalysis_air_temp_k']

fig, ax = plt.subplots()
sns.lineplot(data=T, x=T.index, y=yg, label="label")
sns.lineplot(data=T, x=T.index, y=xg, label="feature")
ax.legend()
fig.show()

Index(['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm'],
      dtype='object')



# sum of precipitation_amt_mm

