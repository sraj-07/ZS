# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 13:47:57 2018

@author: D
"""

import pandas as pd
import numpy as np
train = pd.read_csv("C:/Users/D/Desktop/dataset/yds_train2018.csv")
train.head()
for i in range(len(train['Sales'])):
    if train['Sales'][i]<0:
        train['Sales'][i]=abs(train['Sales'][i])
train_grp = train.groupby(['Product_ID','Country','Year','Month']).agg({'Sales':'sum'})
test = pd.read_csv("C:/Users/D/Desktop/dataset/yds_test2018.csv")
train_grp.reset_index(inplace=True)


holidays = pd.read_csv("C:/Users/D/Desktop/dataset/holidays1.csv")
hol = holidays.groupby(['Country','Year','Month']).agg({'Day':'count'})
hol.reset_index(inplace=True)
expense = pd.read_csv("C:/Users/D/Desktop/dataset/promotional_expense.csv")
#currency = pd.read_csv("C:/Users/D/Desktop/dataset/Currency.csv")

train_grp = pd.merge(train_grp,expense, on=['Product_ID','Country','Month','Year'],how = "left")
train_grp['Expense_Price'].fillna(0.0,inplace=True)
train_grp = pd.merge(train_grp,hol,on = ['Country','Year','Month'],how="left")
train_grp['Day'].fillna(0.0,inplace=True)

#train_grp = pd.merge(train_grp,currency,on=['Country'],how="left")

test = pd.merge(test,expense, on=['Product_ID','Country','Month','Year'],how="left")
test['Expense_Price'].fillna(0.0,inplace=True)
test = pd.merge(test,hol,on = ['Country','Year','Month'],how="left")
test['Day'].fillna(0.0,inplace=True)

#test = pd.merge(test,currency,on=['Country'],how="left")


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(train_grp.Country)
train_grp.Country = le.transform(train_grp.Country)
test.Country = le.transform(test.Country)

a = train_grp['Product_ID'].value_counts().index
a = list(a)

dfs = []
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV
for i in a:
    df = train_grp[train_grp['Product_ID']==i]
    df1 = test[test['Product_ID']==i]
    #regr = RandomForestRegressor(n_estimators=100)

    clf = XGBRegressor(eval_metric = 'rmse',
        nthread = 4,
        eta = 0.1,
        num_boost_round = 80,
        max_depth = 5,
        subsample = 0.5,
        colsample_bytree = 1.0,
        silent = 1,
        )
    parameters = {
        'n_estimators': [10,20,50,100,360,450],
        'num_boost_round': [10, 15,25, 50],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [3, 4, 5],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
    }


    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
    
    
    X = df[['Year','Month','Country','Expense_Price']]
    #X = sm.add_constant(X)
    Y = df['Sales']
    model = clf.fit(X,Y)
    #result = sm.OLS(endog=Y, exog=X, missing='drop').fit()
    #models.append(result)
    y = df1[['Year','Month','Country','Expense_Price']]
    #y = sm.add_constant(y)
    
    #trust your CV!
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    y_pred = model.predict(y)
    
    df1['Sales'] = y_pred
    
    #df1.Country = le.inverse_transform(test.Country)
    
    df1.drop('Expense_Price',axis=1,inplace=True)
    
    df1.drop('Day',axis=1,inplace=True)
    
    #df1.drop('Currency',axis=1,inplace=True)
    
    dfs.append(df1)
    
for i in range(0,5):
    if i <1:
        b = dfs[i]
    else:
        b = pd.concat([b,dfs[i]])
        
x = b.sort_values(['S_No'], ascending=[True])

x.Country = le.inverse_transform(test.Country)

x.to_csv("C:/Users/D/Desktop/dataset/19th_submission.csv", encoding='utf-8', index=False)


