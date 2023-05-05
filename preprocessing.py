import pandas as pd
import numpy as np

def preprocessing(features, test_features):
    
    number_train_features = features.shape[0]

    # Merge the datasets before cleaning:
    all_features = pd.merge(features, test_features, how='outer')

    # Drop week_start_date
    all_features.drop('week_start_date', axis=1, inplace=True)

    # One hot encoding for 'city'
    all_features = pd.merge(pd.get_dummies(all_features.city), all_features.drop('city', axis=1), left_index=True, right_index=True)

    # Return the separated datasets
    return all_features.iloc[:number_train_features], all_features.iloc[number_train_features:]
