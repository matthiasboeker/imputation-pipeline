# Analytical Imputation Pipeline
The imputation pipeline serves to purpose to gain insight and better understanding on
the impact of different imputation algorithms on the data and finally machine learning algorithms
and their performance.

## Installation
```
Not defined yet
```

## Usage
```
import ...
```

## Todo
Is the pipeline defined for a specific data type, like time series? Or generalisable?

Which analytical tools will we provide?

### Access
How can we access the pipeline?

### Imputation Methods
Are the imputation methods specialised for time series?
- KNN Imputation
- Forward/ Backward Fill
- Simple Imputation of Mean or Median
- Singular Spectrum Imputation
- Regression based Imputation
- Spline imputation
- Matrix Completion
- AutoEncoder Approach
- Multiple Imputation/ MICE


### Analytical Tools
How do we want to analyse the impact of imputation methods? What are potential impacts?
- Distribution Skewness
- Change in Distribution (Stat. Test)
- Changed Feature Importance
- Forecasting Bias
- Forecasting Variance
- Test on Algorithm Performance
- Cluster Visualisation with Missing label

### Preprocessing Tools
Time series specific preprocessing
- Test for Stationary
- Box-Cox, Log, Shift, De-Mean transformation
- rolling window operations

### Integration
Allow a lean integration with other frameworks
- Transform to Pandas
- Transform to Tsai
- Transform to Pytorch
