### What is linear regression?

Linear regression is a linear approach to model the relationship between the response variable and the explanatory variables. The expectation of Y given X is linear in the parameters. 	

### Ordinary Least Squares (OLS)

OLS is a kind of methods to estimate the unknown parameters in linear regression. The OLS solution is the parameters which minimize the sum of squared error, which measures the average lack of fit.

The OLS solution is the best linear unbiased estimator 'BLUE'. However, there may exist a biased estimator with smaller MSE, which trades a little bias for a large reduction of variance.

#### What’s the assumption made in OLS?

* Linear structure. The response is a linear combination of the coefficients and the predictors.

* Errors are i.i.d with mean 0 and constant variance. 
  * If correlated error, try the generalized least squares. 
  * If error with inconstant variance, try weighted least squares or do data transformation. For y, try Box-cox transformation

* The independence of the features. 
  * If the predictors are linear dependent, the design matrix becomes singular so that the least square estimate becomes highly sensitive to random errors. This’s called multicollinearity.

