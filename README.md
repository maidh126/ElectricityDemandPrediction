# Approach

- load Pandas DataFrame containing electricity data
- split the data in train, test and validation sets (+ normalise independent variables if required) 
- fit model parameters using GridSearchCV [scikit-learn](http://scikit-learn.org/stable/)
- evaluate estimator performance by means of 5 fold 'shuffled' nested cross-validation
- predict cross validated estimates of y for each data point and plot on scatter diagram vs true y
- find the best model and fit to validation set to find electricity demand


# Packages required

- [Python 3.8](https://www.python.org/downloads/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://docs.scipy.org/doc/)
- [scikit-learn](http://scikit-learn.org/stable/)


## Scores (5 fold nested 'shuffled'cross-validation - Rsquared)

**1. XGBoost Regression**											                                            
  * Parameters: 
  * Score: 

**2. Random Forest Regression**        									                                   
  * Parameters: 
  * Score:

**3. Polynomial Regression**                                							
  * Parameters: 
  * Score:
  
**4. Neural Network MLP Regression** 				     					
  * Parameters: 
  * Score: 
  
**5. KNN Regression**                                   							
  * Parameters:
  * Score: 

**6. Ordinary Least-Squares Regression**                                    				
  * Parameters: 
  * Score: 
  
**7. Ridge Regression** 		                                        				
  * Parameters: 
  * Score: 
  
**8. Lasso Regression**                                        	 					
  * Parameters: 
  * Score: 

## Sample data input (Pandas DataFrame)

```
       period	temperature	  hours before sunrise	   hours before sunset	  demand
0	  1	        8.4	            6.016667	              17.633333	      496.0
1	  2	        8.1	            5.516667	              17.133333	      535.0
2	  3	        7.8	            5.016667	              16.633333	      511.0
3	  4	        7.5	            4.516667	              16.133333	      496.0
4	  5	        7.3	            4.016667	              15.633333	      490.0
```

#### 1. Linear Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/master/1_linear Regression.png)

#### 2. XGBoost

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/master/2. XGBoost.png)

#### 3. KNN

![alt text](https://github.com/maidh126/ElectricityDemandPrediction/blob/main/plot/3. KNN.png)

#### 4. Polynomial

![alt text](https://github.com/maidh126/ElectricityDemandPrediction//blob/main/plot/4. Lasso.png)

#### 5. Ridge

![alt text](https://github.com/maidh126/ElectricityDemandPrediction//blob/main/plot/5. Ridge.png)

#### 6. Polynomial Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction//blob/main/plot/6. Polynomial Regression.png)

#### 7. Decision Tree Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction//blob/main/plot/7. Decision Tree Regression.png)

#### 8. Random Forest

![alt text](https://github.com/maidh126/ElectricityDemandPrediction//blob/main/plot/8. Random Forest.png)

#### 9. MLP Regression

![alt text](https://github.com/maidh126/ElectricityDemandPrediction//blob/main/plot/9. MLP Regression.png)



# References
https://medium.com/analytics-vidhya/scikit-learn-pipeline-d43c80559257
https://github.com/isheunesutembo/Scikit-Learn-Pipelines/blob/master/SkLearn%20Pipelines.ipynb
