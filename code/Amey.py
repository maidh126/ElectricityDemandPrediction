#!/usr/bin/env python
# coding: utf-8

# ## Approach

# - load Pandas DataFrame containing electricity data
# - split the data in train, test and validation sets (+ normalise independent variables if required)
# - fit model parameters using GridSearchCV [scikit-learn](http://scikit-learn.org/stable/)
# - evaluate estimator performance by means of 5 fold 'shuffled' nested cross-validation
# - predict cross validated estimates of y for each data point and plot on scatter diagram vs true y
# - find the best model and fit to validation set to find electricity demand

# ## Packages required

# - [Python 3.8](https://www.python.org/downloads/)
# - [Matplotlib](https://matplotlib.org/)
# - [Pandas](https://pandas.pydata.org/)
# - [Numpy](https://docs.scipy.org/doc/)
# - [scikit-learn](http://scikit-learn.org/stable/)

# ## Implement 

# #### Install packages

# In[1]:


get_ipython().system('pip install scikit-learn')
# !pip install xgboost


# In[2]:


import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
# import xgboost as xgb


# #### Preprocessing

# - Read the dataset

# In[16]:


file = pd.read_csv('Data.csv')
file = pd.DataFrame(file)
file


# - Split to train, test and validation datasets

# In[17]:


df = file[file.demand.notnull()]
df


# In[18]:


y = df.demand
y


# - Drop period and demand

# In[19]:


X = df.drop('period', axis=1).drop('demand', axis = 1)
X


# - Train/test split

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# - Validation set

# In[21]:


df_vali = file[file.demand.isnull()]


# In[22]:


vali = df_vali.drop('period', axis=1).drop('demand', axis = 1)
vali


# #### Defind the pipeline models

# - defind pipeline
# - cross validation
# - show model coefficients or feature importances
# - plot predicted demand vs actual demand
# - fit the validation set

# In[35]:


def model(pipeline, parameters, X_train, y_train, X, y):

    grid_obj = GridSearchCV(estimator=pipeline,
                            param_grid=parameters,
                            cv=3,
                            scoring='r2',
                            verbose=2,
                            n_jobs=1,
                            refit=True)
    grid_obj.fit(X_train, y_train)
    
    grid_obj.predict(vali)
    
    
    '''Results'''

    results = pd.DataFrame(pd.DataFrame(grid_obj.cv_results_))
#     results_sorted = results.sort_values(by=['mean_test_score'], ascending=False)
    results_vali = grid_obj.predict(vali)
    

    print("##### Results")
#     print(results_sorted)
    print(results)
    print(results_vali)

    print("best_index", grid_obj.best_index_)
    print("best_score", grid_obj.best_score_)
    print("best_params", grid_obj.best_params_)

    
    '''Cross Validation'''
    # Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

    estimator = grid_obj.best_estimator_
    '''
    if estimator.named_steps['scl'] == True:
        X = (X - X.mean()) / (X.std())
        y = (y - y.mean()) / (y.std())
    '''
    shuffle = KFold(n_splits=5,
                    shuffle=True,
                    random_state=0)
    cv_scores = cross_val_score(estimator,
                                X,
                                y.values.ravel(),
                                cv=shuffle,
                                scoring='r2')
    print("##### CV Results")
    print("mean_score", cv_scores.mean())

    
    '''Show model coefficients or feature importances''' 
    # Feature importance refers to how useful a feature is at predicting a target variable. 
    # A coefficient refers to a number or quantity placed with a variable.

    try:
        print("Model coefficients: ", list(zip(list(X), estimator.named_steps['clf'].coef_)))
    except:
        print("Model does not support model coefficients")

    try:
        print("Feature importances: ", list(zip(list(X), estimator.named_steps['clf'].feature_importances_)))
    except:
        print("Model does not support feature importances")

        
    '''Predict y vs y_predicted in scatter'''

    y_pred = cross_val_predict(estimator, X, y, cv=shuffle)

    plt.scatter(y, y_pred)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], "g--", lw=1, alpha=0.4)
    plt.xlabel("True demand")
    plt.ylabel("Predicted demand")
    plt.annotate(' R-squared CV = {}'.format(round(float(cv_scores.mean()), 3)), size=9,
             xy=(xmin,ymax), xytext=(10, -15), textcoords='offset points')
    plt.annotate(grid_obj.best_params_, size=9,
                 xy=(xmin, ymax), xytext=(10, -35), textcoords='offset points', wrap=True)
    plt.title('predicted demand vs actual demand')
    plt.show()

    
    '''Fit the validation set'''
    
    # convert array to serial 
    vali_series = pd.Series(results_vali)

    df_vali.iloc[:,4] = vali_series.values
    print(df_vali)


# #### Pipeline and Parameters

# - Linear Regression

# In[36]:


pipe_ols = Pipeline([('scl', StandardScaler()),
           ('clf', LinearRegression())])

param_ols = {}


# - XGBoost

# In[37]:


# # - XGBoost

# pipe_xgb = Pipeline([('clf', xgb.XGBRegressor())])

# param_xgb = {'clf__max_depth':[5],
#              'clf__min_child_weight':[6],
#              'clf__gamma':[0.01],
#              'clf__subsample':[0.7],
#              'clf__colsample_bytree':[1]}


# - KNN

# In[38]:


pipe_knn = Pipeline([('clf', KNeighborsRegressor())])

param_knn = {'clf__n_neighbors':[5, 10, 15, 25, 30]}


# - Lasso

# In[39]:


pipe_lasso = Pipeline([('scl', StandardScaler()),
                       ('clf', Lasso(max_iter=1500))])

param_lasso = {'clf__alpha': [0.01, 0.1, 1, 10]}


# - Ridge

# In[40]:


pipe_ridge = Pipeline([('scl', StandardScaler()),
                       ('clf', Ridge())])

param_ridge = {'clf__alpha': [0.01, 0.1, 1, 10]}


# - Polynomial Regression

# In[41]:


pipe_poly = Pipeline([('scl', StandardScaler()),
                      ('polynomial', PolynomialFeatures()),
                      ('clf', LinearRegression())])

param_poly = {'polynomial__degree': [2, 4, 6]}


# - Decision Tree Regression

# In[42]:


pipe_tree = Pipeline([('clf', DecisionTreeRegressor())])

param_tree = {'clf__max_depth': [2, 5, 10],
             'clf__min_samples_leaf': [5,10,50,100]}


# - Random Forest

# In[43]:


pipe_forest = Pipeline([('clf', RandomForestRegressor())])

param_forest = {'clf__n_estimators': [10, 20, 50],
                'clf__max_features': [None, 1, 2],
                'clf__max_depth': [1, 2, 5]}


# - MLP Regression

# In[44]:


pipe_neural = Pipeline([('scl', StandardScaler()),
                        ('clf', MLPRegressor())])

param_neural = {'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'clf__hidden_layer_sizes': [(5),(10,10),(7,7,7)],
                'clf__solver': ['lbfgs'],
                'clf__activation': ['relu', 'tanh'],
                'clf__learning_rate' : ['constant', 'invscaling']}


# #### Execute model hyperparameter tuning and crossvalidation

# - Linear Regression

# In[45]:


model(pipe_ols, param_ols, X_train, y_train, X, y)


# - XGBoost

# In[46]:


# model(pipe_xgb, param_xgb, X_train, y_train, X, y)


# - KNN

# In[47]:


model(pipe_knn, param_knn, X_train, y_train, X, y)


# - Lasso

# In[48]:


model(pipe_lasso, param_lasso, X_train, y_train, X, y)


# - Ridge

# In[49]:


model(pipe_ridge, param_ridge, X_train, y_train, X, y)


# - Polynomial Regression

# In[50]:


model(pipe_poly, param_poly, X_train, y_train, X, y)


# - Decision Tree Regression

# In[51]:


model(pipe_tree, param_tree, X_train, y_train, X, y)


# - Random Forest

# In[52]:


model(pipe_forest, param_forest, X_train, y_train, X, y)


# - Multi-layer Perceptron (MLP) Regression

# In[53]:


model(pipe_neural, param_neural, X_train, y_train, X, y)


# ## Conclusion

# **KNN fits the best**
# 
# **1. KNN**                            							
#   * Parameters: clf__n_neighbors: 25
#   * Score: 0.918
#   
# **2. Decision Tree Regression**		                                        				
#   * Parameters: clf__max_depth: 10, clf__min_samples_leaf: 10
#   * Score: 0.911
# 
# **3. Polynomial Regression**                                    				
#   * Parameters: polynomial__degree: 6
#   * Score: 0.87
# 
# **4. Random Forest**                                        	 					
#   * Parameters: clf__max_depth: 5, clf__max_features: 2, clf__n_estimators: 50
#   * Score: 0.883
# 
# **5. Linear Regression**											                                            
#   * Parameters: non
#   * Score: 0.548
#   
# **6. Lasso** 				     					
#   * Parameters: clf__alpha: 0.01
#   * Score: 0.548
#   
# **7. Ridge**                                   							
#   * Parameters: clf__alpha: 1
#   * Score: 0.548
# 
# **8. XGBoost**        									                                   
#   * Parameters: clf_colsample_bytree: 1, clf_gamma: 0.01, clf_ max_depth: 5, clf_min_child_weight': 6, clf_subsample: 0.7
#   * Score: 0.918
#   
# **9. Multi-layer Perceptron (MLP) Regression**                                        	 					
#   * Parameters: 'clf__activation': 'relu', 'clf__alpha': 0.001, 'clf__hidden_layer_sizes': (7, 7, 7), 'clf__learning_rate': 'constant', 'clf__solver': 'lbfgs'
#   * Score: 0.862

# ## References

# - [Pipeline](https://medium.com/analytics-vidhya/scikit-learn-pipeline-d43c80559257) and [Github](https://github.com/isheunesutembo/Scikit-Learn-Pipelines/blob/master/SkLearn%20Pipelines.ipynb)
