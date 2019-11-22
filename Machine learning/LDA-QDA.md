To find the linear decision boundary, we need to know the class posteriors Pr(G|X) for optimal classification. Let pk be the prior probability of class k, with sum(pi) = 1 and fk(x) is the probability of x under class k. So, by Bayes rule, Pr(G = k| X = x) ~ pk * fk(x).

We estimate pk by the ratio of (# of class k) and (total # of samples).

* How to estimate fk(x)?
  * LDA and QDA use Gaussian densities.
  * Other methods.

### Linear discriminant analysis (LDA)

For LDA, we model each class density as multivariate Gaussian with a common covariance matrix. The estimated parameters are following:
•	Miu_k = sample mean for data in class k
•	Covariance = weighted sample covariance.

### Quadratic discriminant analysis (QDA) 

Separate covariance matrix will be estimated for each class. When p is large, this means a dramatic increase in parameters.

#### Difference between LDA and logistic regression:
- The linearity of LDA comes from the Gaussian assumption on the class and common covariance matrix. But the linearity is constructed by the logit model.
- Logistic Regression maximizes the conditional likelihood Pr(G|X)
- LDA maximizes the full-likelihood based on the joint density. 
- LDA is not robust to outliers because the outlier plays a role in estimating the covariance matrix. But logistic regression is more robust.
- The maximum likelihood estimates of the parameters are not defined when the two class can be perfectly separated in logistic regression.
- Because the assumptions are never correct in practice, logistic regression is safer and more robust than LDA. But the LDA provides similar results even the assumptions are not satisfied. 
