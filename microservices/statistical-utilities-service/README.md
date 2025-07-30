# Statistical Utilities Service

A comprehensive microservice for general statistical computations including descriptive statistics, hypothesis testing, correlation analysis, and more.

## Features

- **Descriptive Statistics**: Mean, median, mode, variance, skewness, kurtosis
- **Hypothesis Testing**: t-tests, chi-square tests, ANOVA, non-parametric tests
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations
- **Resampling Methods**: Bootstrap, jackknife, permutation tests
- **Distribution Fitting**: Fit data to various probability distributions
- **Time Series Analysis**: Autocorrelation, trend analysis, seasonality
- **Multivariate Analysis**: PCA, factor analysis, canonical correlation

## API Endpoints

### Health Check
```http
GET /health
```

### Descriptive Statistics
```http
POST /stats/summary
```
Request body:
```json
{
  "data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "statistics": ["mean", "median", "std", "skewness", "kurtosis"]
}
```

### Hypothesis Testing
```http
POST /stats/hypothesis-test
```
Request body:
```json
{
  "test": "t-test",
  "type": "two-sample",
  "data1": [1.0, 2.0, 3.0],
  "data2": [4.0, 5.0, 6.0],
  "alternative": "two-sided",
  "alpha": 0.05
}
```

### Correlation Analysis
```http
POST /stats/correlation
```
Request body:
```json
{
  "x": [1.0, 2.0, 3.0, 4.0, 5.0],
  "y": [2.0, 4.0, 5.0, 4.0, 5.0],
  "method": "pearson",
  "confidence_level": 0.95
}
```

### Bootstrap Resampling
```http
POST /stats/resample
```
Request body:
```json
{
  "data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "method": "bootstrap",
  "n_samples": 1000,
  "statistic": "mean",
  "confidence_level": 0.95
}
```

### Distribution Fitting
```http
POST /stats/fit-distribution
```
Request body:
```json
{
  "data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "distributions": ["normal", "exponential", "gamma"],
  "method": "mle"
}
```

### Time Series Analysis
```http
POST /stats/time-series
```
Request body:
```json
{
  "data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "analysis": ["acf", "pacf", "trend", "seasonality"],
  "lags": 10
}
```

### Principal Component Analysis
```http
POST /stats/pca
```
Request body:
```json
{
  "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  "n_components": 2,
  "standardize": true
}
```

### ECDF (Empirical Cumulative Distribution Function)
```http
POST /stats/ecdf
```
Request body:
```json
{
  "data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "eval_points": [1.5, 2.5, 3.5]
}
```

## Building and Running

### Local Build
```bash
mkdir build
cd build
cmake ..
make
./statistical_utilities_service 0.0.0.0 8089
```

### Docker
```bash
docker build -t statistical-utilities-service .
docker run -p 8089:8089 statistical-utilities-service
```

## Examples

### Get Summary Statistics
```bash
curl -X POST http://localhost:8089/stats/summary \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "statistics": ["mean", "median", "std", "min", "max"]
  }'
```

### Perform t-test
```bash
curl -X POST http://localhost:8089/stats/hypothesis-test \
  -H "Content-Type: application/json" \
  -d '{
    "test": "t-test",
    "type": "one-sample",
    "data1": [1, 2, 3, 4, 5],
    "mu": 3.0,
    "alternative": "two-sided"
  }'
```

### Calculate Correlation
```bash
curl -X POST http://localhost:8089/stats/correlation \
  -H "Content-Type: application/json" \
  -d '{
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 5, 4, 5],
    "method": "pearson"
  }'
```

## Dependencies

- Boost.Beast (HTTP server)
- Eigen3 (Linear algebra)
- C++17 compiler