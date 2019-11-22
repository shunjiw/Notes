### What is linear regression?

Linear regression is a linear approach to model the relationship between the response variable and the explanatory variables. The expectation of Y given X is linear in the parameters. 	

### Ordinary Least Squares (OLS)

OLS is a kind of methods to estimate the unknown parameters in linear regression. The OLS solution is the parameters which minimize the sum of squared error, which measures the average lack of fit.

The OLS solution is the best linear unbiased estimator 'BLUE'. However, there may exist a biased estimator with smaller MSE, which trades a little bias for a large reduction of variance.

#### What’s the assumption made in OLS?

* Linear structure. The response is a linear combination of the coefficients and the predictors.
* Errors are i.i.d with mean 0 and constant variance. 
  - If correlated error, try the generalized least squares. If error with inconstant variance, try weighted least squares or do data transformation. For y, try Box-cox transformation
* The independence of the features. 
  - If the predictors are linear dependent, the design matrix becomes singular so that the least square estimate becomes highly sensitive to random errors. This’s called multicollinearity.

### Ridge Regression

Ridge regreesion minimizes the penalized residual sum of squares: MSE + lambda * l2_norm of coefficients. Lambda is the complexity parameter which controls the amount of shrinkage. The larger the lambda, the larger the amount of the shrinkage. 

When there are collinearities in the linear regression, the coefficients are poorly fitted and show high variance. But shrinkage method can solve this problem. Need to standardize the input and the intercept term is out of the penalty.

### Lasso 

The Lasso is a linear model that estimates sparse coefficients, which minimizes the penalized residual sum of squares: MSE + lambda * l1_norm of coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number of features upon which the given solution is dependent. 

* How to set lambda in Lasso? Cross validation or information criteria (AIC or BIC)

### Logistic (Logit) Regression

Logistic regression is a linear model for classification. It applies to binary, one to rest, and multiple cases. For k class, model the log-odds by the linear function of x. By default, the last one is the reference class. 

The models are fit by the maximum likelihood. Because we model the conditional probability Pr(G|X), the multinomial distribution is appropriate. The algorithm to solve logistic regression is Newton-Raphson algorithm or iteratively reweighted least squares (IRLS). 

### Poisson Regression

See details on https://en.wikipedia.org/wiki/Poisson_regression
