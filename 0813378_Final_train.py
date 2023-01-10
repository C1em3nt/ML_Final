import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import HuberRegressor

from sklearn.linear_model import LogisticRegression


from category_encoders import WOEEncoder
import matplotlib.pyplot as plt

train = pd.read_csv('tabular-playground-series-aug-2022/train.csv')
test = pd.read_csv('tabular-playground-series-aug-2022/test.csv')
submission = pd.read_csv('tabular-playground-series-aug-2022/sample_submission.csv')
y = train['failure']
train.drop('failure',axis=1, inplace = True)

def prepreprocessing(train, test):
    data = pd.concat([train, test])
    data['m3_missing'] = data['measurement_3'].isna().astype(np.int8)
    data['m5_missing'] = data['measurement_5'].isna().astype(np.int8)
    data['missing(3*5)'] = data['m5_missing'] * (data['m3_missing'])
    
    #data['mix'] = data['attribute_2'] * data['attribute_3']
    features =[]
    for feat in test.columns:
        if feat.startswith('measurement') or feat=='loading':
            features.append(feat)
    #print(features)

    full_impute ={}
    
    cols =[]
    for col in test.columns:
        if 'measurement' not in col:
            cols.append(col)
    cols += ['loading','m3_missing','m5_missing']
    #print(cols)
    
    most_4corr = []
    meas =[]
    for x in range(3,18):   
        corr = np.absolute(data.drop(cols, axis=1).corr()[f'measurement_{x}']).sort_values(ascending=False)
        most_4corr.append(np.round(np.sum(corr[1:5]),5))
        meas.append(f'measurement_{x}')
        
    tot_corr = pd.DataFrame()
    tot_corr['measurements'] = meas
    tot_corr['tot_correlation'] = most_4corr
    tot_corr = tot_corr.sort_values(by ='tot_correlation',ascending=False).reset_index(drop = True)

    #print(c.iloc[0,0][:])
    #first 4 correlation between two measurements(first 8 in c) in each product_codes 
    for i in range(8):
        measurement_col = tot_corr.iloc[i,0][:] 
        impute ={}
        for x in data.product_code.unique() : 
            corr = np.absolute(data[data.product_code == x].drop(cols, axis=1).corr()[measurement_col]).sort_values(ascending=False)
            measurement = {}
            measurement[measurement_col] = corr[1:5].index.tolist()
            impute[x] = measurement[measurement_col]
        full_impute[measurement_col] =impute
    #print(full_impute)
    
    features =[]
    for feat in data.columns:
        if feat.startswith('measurement') or feat=='loading':
            features.append(feat)
            
    null_cols = []
    for col in train.columns:
        if train[col].isnull().sum()!=0:
            null_cols.append(col)
            
    for code in data.product_code.unique():
        filled_by_Huber = 0
        for measurement_col in list(full_impute.keys()):
            tmp = data[data.product_code==code]
            column = full_impute[measurement_col][code]
            tmp_train = tmp[column+[measurement_col]].dropna(how='any')
            tmp_test = tmp[(tmp[column].isnull().sum(axis=1)==0)&(tmp[measurement_col].isnull())]
            #print(tmp_test)
            model = HuberRegressor(epsilon=1.45, max_iter = 400)
            model.fit(tmp_train[column], tmp_train[measurement_col])
            data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data[measurement_col].isnull()),measurement_col] = model.predict(tmp_test[column])
            filled_by_Huber += len(tmp_test)

        model1 = KNNImputer(n_neighbors=3)
        data.loc[data.product_code==code,features] = model1.fit_transform(data.loc[data.product_code==code, features])
    
    return data
    
data = prepreprocessing(train, test)

select_feature = ['loading',
                  'attribute_0',
                  'attribute_1',
                  'measurement_17',
                  #'measurement_0',
                  #'measurement_1',
                  #'mix',
                  #'measurement_2',
                  #'measurement_11',
                  'missing(3*5)',
                  'm3_missing',
                  'm5_missing',
                  ]

train = data.iloc[:train.shape[0],:]
test = data.iloc[train.shape[0]:,:]
#print(train.shape, test.shape)
groups = train.product_code
X = train

woe_encoder = WOEEncoder(cols=['attribute_0'])
woe_encoder.fit(X, y)
X = woe_encoder.transform(X)
test = woe_encoder.transform(test)

woe_encoder = WOEEncoder(cols=['attribute_1'])
woe_encoder.fit(X, y)
X = woe_encoder.transform(X)
test = woe_encoder.transform(test)

def scale(train_data, val_data, test_data, features):
    scaler = StandardScaler()

    scaled_train = scaler.fit_transform(train_data[features])
    scaled_val = scaler.transform(val_data[features])
    scaled_test = scaler.transform(test_data[features])
    
    train = train_data.copy()
    val = val_data.copy()
    test = test_data.copy()
    
    train[features] = scaled_train
    val[features] = scaled_val
    test[features] = scaled_test
    
    return train, val, test

lr_test = np.zeros(len(test))
x_train, x_val, x_test = scale(X[select_feature], X[select_feature], test, select_feature)
model = LogisticRegression(max_iter=1000, C=0.0001, penalty='l2', solver='newton-cg')
model.fit(x_train[select_feature], y)

filename = "my_model.pickle"
pickle.dump(model, open(filename, "wb"))

print("Model Saved.")
