## Approach

- load Pandas DataFrame containing electricity data
- split the data in train, test and validation sets (+ normalise independent variables if required) 
- fit model parameters using GridSearchCV [scikit-learn](http://scikit-learn.org/stable/)
- evaluate estimator performance by means of 5 fold 'shuffled' nested cross-validation
- predict cross validated estimates of y for each data point and plot on scatter diagram vs true y
- find the best model and fit to validation set to find missing electricity demand


## Packages required

- [Python 3.8](https://www.python.org/downloads/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://docs.scipy.org/doc/)
- [scikit-learn](http://scikit-learn.org/stable/)


## Sample data input (Pandas DataFrame)

```
       period	    temperature            hours before sunrise	  hours before sunset      demand
0	  1	        8.4	            6.016667	              17.633333	      496.0
1	  2	        8.1	            5.516667	              17.133333	      535.0
2	  3	        7.8	            5.016667	              16.633333	      511.0
3	  4	        7.5	            4.516667	              16.133333	      496.0
4	  5	        7.3	            4.016667	              15.633333	      490.0
```

## Scores (5 fold nested 'shuffled'cross-validation - Rsquared)

**1. Linear Regression**											                                            
  * Parameters: non
  * Score: 0.548

**3. KNN**                                							
  * Parameters: clf__n_neighbors: 25
  * Score: 0.918
  
**4. Lasso** 				     					
  * Parameters: clf__alpha: 0.01
  * Score: 0.548
  
**5. Ridge**                                   							
  * Parameters: clf__alpha: 1
  * Score: 0.548

**6. Polynomial Regression**                                    				
  * Parameters: polynomial__degree: 6
  * Score: 0.87
  
**7. Decision Tree Regression** 		                                        				
  * Parameters: clf__max_depth: 10, clf__min_samples_leaf: 10
  * Score: 0.911
  
**8. Random Forest**                                        	 					
  * Parameters: clf__max_depth: 5, clf__max_features: 2, clf__n_estimators: 50
  * Score: 0.883

**9. Multi-layer Perceptron (MLP) Regression**                                        	 					
  * Parameters: 'clf__activation': 'relu', 'clf__alpha': 0.001, 'clf__hidden_layer_sizes': (7, 7, 7), 'clf__learning_rate': 'constant', 'clf__solver': 'lbfgs'
  * Score: 0.862


#### 1. Linear Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/1_LinearRegression.png)

#### 3. KNN

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/3_KNN.png)

#### 4. Polynomial

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/4_Lasso.png)

#### 5. Ridge

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/5_Ridge.png)

#### 6. Polynomial Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/6_PolynomialRegression.png)

#### 7. Decision Tree Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/7_DecisionTreeRegression.png)

#### 8. Random Forest

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/8_RandomForest.png)

#### 9. MLP Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plots/9_MLPRegression.png)



## Conclusion
KNN fits the best

## Apply model to validation set
```
       period  temperature  hours before sunrise  hours before sunset   demand
48240   48241         11.9              3.833333            20.316667   530.16
48241   48242         12.0              3.333333            19.816667   517.68
48242   48243         12.1              2.833333            19.316667   487.36
48243   48244         12.0              2.333333            18.816667   477.96
48244   48245         11.9              1.833333            18.316667   468.40
...       ...          ...                   ...                  ...      ...
52555   52556         12.4            -15.516667            -3.800000  1156.68
52556   52557         12.3            -16.016667            -4.300000  1051.48
52557   52558         12.2            -16.516667            -4.800000   846.08
52558   52559         11.9            -17.016667            -5.300000   662.60
52559   52560         11.9            -17.516667            -5.800000   591.56
```


# References
- [Pipeline](https://medium.com/analytics-vidhya/scikit-learn-pipeline-d43c80559257) and [Github](https://github.com/isheunesutembo/Scikit-Learn-Pipelines/blob/master/SkLearn%20Pipelines.ipynb)
