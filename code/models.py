import warnings

warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')

!pip
install
scikit - learn
# !pip install xgboost

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

# read csv
file = pd.read_csv('Data.csv')
file = pd.DataFrame(file)

## split to (train + test) and validation datasets
# train + test dataset
df = file[file.demand.notnull()]
df

# y demand
y = df.demand
y

# drop period (index?) and demand
X = df.drop('period', axis=1).drop('demand', axis=1)
X

# execute preprocessing & train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# validation datasets
df_vali = file[file.demand.isnull()]
vali = df_vali.drop('period', axis=1).drop('demand', axis=1)
vali


## defind model for models =))
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
    # results_sorted = results.sort_values(by=['mean_test_score'], ascending=False)
    results_vali = grid_obj.predict(vali)

    print("##### Results")
    # print(results_sorted)
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

    '''Predict along CV and plot y vs. y_predicted in scatter'''

    y_pred = cross_val_predict(estimator, X, y, cv=shuffle)

    plt.scatter(y, y_pred)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], "g--", lw=1, alpha=0.4)
    plt.xlabel("True demand")
    plt.ylabel("Predicted demand")
    plt.annotate(' R-squared CV = {}'.format(round(float(cv_scores.mean()), 3)), size=9,
                 xy=(xmin, ymax), xytext=(10, -15), textcoords='offset points')
    plt.annotate(grid_obj.best_params_, size=9,
                 xy=(xmin, ymax), xytext=(10, -35), textcoords='offset points', wrap=True)
    plt.title('Predicted demand vs. demand')
    plt.show()

    ## print results to valiadation set
    # convert array to serial
    new_series = pd.Series(results_vali)

    df_vali.iloc[:, 4] = new_series.values
    print(df_vali)


## scikit learn models

# 1. Pipeline and Parameters - Linear Regression

pipe_ols = Pipeline([('scl', StandardScaler()),
                     ('clf', LinearRegression())])

param_ols = {}

# 2. Pipeline and Parameters - XGBoost

pipe_xgb = Pipeline([('clf', xgb.XGBRegressor())])

param_xgb = {'clf__max_depth': [5],
             'clf__min_child_weight': [6],
             'clf__gamma': [0.01],
             'clf__subsample': [0.7],
             'clf__colsample_bytree': [1]}

# 3. Pipeline and Parameters - KNN

pipe_knn = Pipeline([('clf', KNeighborsRegressor())])

param_knn = {'clf__n_neighbors': [5, 10, 15, 25, 30]}

# 4. Pipeline and Parameters - Lasso

pipe_lasso = Pipeline([('scl', StandardScaler()),
                       ('clf', Lasso(max_iter=1500))])

param_lasso = {'clf__alpha': [0.01, 0.1, 1, 10]}

# 5. Pipeline and Parameters - Ridge

pipe_ridge = Pipeline([('scl', StandardScaler()),
                       ('clf', Ridge())])

param_ridge = {'clf__alpha': [0.01, 0.1, 1, 10]}

# 6. Pipeline and Parameters - Polynomial Regression

pipe_poly = Pipeline([('scl', StandardScaler()),
                      ('polynomial', PolynomialFeatures()),
                      ('clf', LinearRegression())])

param_poly = {'polynomial__degree': [2, 4, 6]}

# 7. Pipeline and Parameters - Decision Tree Regression

pipe_tree = Pipeline([('clf', DecisionTreeRegressor())])

param_tree = {'clf__max_depth': [2, 5, 10],
              'clf__min_samples_leaf': [5, 10, 50, 100]}

# 8. Pipeline and Parameters - Random Forest

pipe_forest = Pipeline([('clf', RandomForestRegressor())])

param_forest = {'clf__n_estimators': [10, 20, 50],
                'clf__max_features': [None, 1, 2],
                'clf__max_depth': [1, 2, 5]}

# 9. Pipeline and Parameters - Multi-layer Perceptron (MLP) Regression

pipe_neural = Pipeline([('scl', StandardScaler()),
                        ('clf', MLPRegressor())])

param_neural = {'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'clf__hidden_layer_sizes': [(5), (10, 10), (7, 7, 7)],
                'clf__solver': ['lbfgs'],
                'clf__activation': ['relu', 'tanh'],
                'clf__learning_rate': ['constant', 'invscaling']}

## execute model hyperparameter tuning and crossvalidation

# 1. Linear Regression
model(pipe_ols, param_ols, X_train, y_train, X, y)

# 2. XGBoost
model(pipe_xgb, param_xgb, X_train, y_train, X, y)

# 3. KNN
model(pipe_knn, param_knn, X_train, y_train, X, y)

# 4. Lasso
model(pipe_lasso, param_lasso, X_train, y_train, X, y)

# 5. Ridge
model(pipe_ridge, param_ridge, X_train, y_train, X, y)

# 6. Polynomial Regression
model(pipe_poly, param_poly, X_train, y_train, X, y)

# 7. Decision Tree Regression
model(pipe_tree, param_tree, X_train, y_train, X, y)

# 8. Random Forest
model(pipe_forest, param_forest, X_train, y_train, X, y)

# 9. Multi-layer Perceptron (MLP) Regression
model(pipe_neural, param_neural, X_train, y_train, X, y)

## Conclusion: KNN fits the best


