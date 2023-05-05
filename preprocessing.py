import pandas as pd
import numpy as np

def preprocessing(features, test_features):
    
    number_train_features = features.shape[0]
    number_test_features = test_features.shape[0]

    # Merge the datasets before cleaning:
    all_features = pd.merge(features, test_features, how='outer')

    # Drop 'week_start_date' and duplicate columns
    all_features.drop(['week_start_date', 'precipitation_amt_mm'], axis=1, inplace=True)
   

    # One hot encoding for 'city'
    all_features = pd.merge(pd.get_dummies(all_features.city), all_features.drop('city', axis=1), left_index=True, right_index=True)

    # Finding all the precipitation columns
    precip_cols = [col for col in all_features.columns if 'precip' in col]
    
    # Adding to each observation the precipation data of the past four weeks
    for i in range(3):
        shifted_columns = all_features[precip_cols].shift(periods=i+1)
        shifted_columns = shifted_columns.add_suffix(str(i+1))
        all_features = pd.concat([all_features, shifted_columns],axis=1)
    
    # Return the separated datasets
    return all_features.iloc[:number_train_features], all_features.iloc[number_train_features:]
