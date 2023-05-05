import pandas as pd
import numpy as np

def preprocessing(features, test_features):
    
    number_train_features = features.shape[0]
    number_test_features = test_features.shape[0]

    # Merge the datasets before cleaning:
    all_features = pd.merge(features, test_features, how='outer')

    # Drop 'year, 'week_start_date' and duplicate column 'precipitation_amt_mm'
    all_features.drop(['year', 'week_start_date', 'precipitation_amt_mm'], axis=1, inplace=True)
   
    # One hot encoding for 'city'
    all_features = pd.merge(pd.get_dummies(all_features.city), all_features.drop('city', axis=1), left_index=True, right_index=True)

    # Defining the columns for adding the data of the past weeks:
    cols = all_features.columns.drop(['iq', 'sj', 'weekofyear'])
    
    # Adding to each observation the data of the past five weeks
    for i in range(5):
        shifted_columns = all_features[cols].shift(periods=i+1)
        shifted_columns = shifted_columns.add_suffix('_'+str(i+1))
        all_features = pd.concat([all_features, shifted_columns],axis=1)
         
        
    # Return the separated datasets
    return all_features.iloc[:number_train_features], all_features.iloc[number_train_features:]
