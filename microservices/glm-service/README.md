# GLM (Generalized Linear Models) Service

A microservice providing comprehensive generalized linear model fitting, prediction, and diagnostics.

## Features

- **Multiple GLM Types**:
  - Linear Regression (Gaussian family, identity link)
  - Logistic Regression (Binomial family, logit link)
  - Poisson Regression (Poisson family, log link)
  - Probit Regression (Binomial family, probit link)

- **Advanced Features**:
  - L1, L2, and Elastic Net regularization
  - Weighted regression
  - Data standardization
  - Automatic intercept handling
  - Prediction intervals

- **Comprehensive Diagnostics**:
  - Multiple residual types (raw, standardized, deviance)
  - Goodness of fit measures (R², adjusted R², AIC, BIC)
  - Influence diagnostics (leverage, Cook's distance)
  - Statistical tests (Durbin-Watson, Breusch-Pagan, Shapiro-Wilk)

## API Endpoints

### Health Check
```
GET /health
```

Returns service status and supported model types.

### Fit Model
```
POST /glm/{model_type}/fit
Content-Type: application/json

{
    "X": [[1.2, 3.4], [5.6, 7.8], ...],  // Feature matrix
    "y": [0, 1, 0, ...],                  // Response vector
    "fit_intercept": true,                // Optional (default: true)
    "standardize": false,                 // Optional (default: false)
    "regularization": 0.1,                // Optional (default: 0.0)
    "regularization_type": "l2",          // Optional: "l1", "l2", "elastic"
    "weights": [1.0, 0.8, ...]           // Optional observation weights
}
```

#### Model Types:
- `linear` or `regression` - Linear regression
- `logistic` - Logistic regression
- `poisson` - Poisson regression
- `probit` - Probit regression

#### Response:
```json
{
    "model_type": "logistic",
    "coefficients": [0.523, -0.234, 0.891],
    "intercept": 1.234,
    "iterations": 7,
    "converged": true,
    "log_likelihood": -234.56,
    "aic": 475.12,
    "bic": 489.34,
    "standard_errors": [0.102, 0.087, 0.145],
    "p_values": [0.0001, 0.023, 0.0003],
    "confidence_intervals": [
        [0.323, 0.723],
        [-0.404, -0.064],
        [0.607, 1.175]
    ]
}
```

### Predict
```
POST /glm/{model_type}/predict
Content-Type: application/json

{
    "X": [[1.2, 3.4], [5.6, 7.8], ...],
    "coefficients": [0.523, -0.234, 0.891],
    "intercept": 1.234,
    "type": "response",               // "response", "linear", "probability"
    "prediction_intervals": true,     // Optional (default: false)
    "alpha": 0.05                    // Optional (default: 0.05)
}
```

#### Prediction Types:
- `response` - Predicted response values (default)
- `linear` - Linear predictor values (before link function)
- `probability` - Predicted probabilities (for logistic/probit only)

#### Response:
```json
{
    "model_type": "logistic",
    "predictions": [0.723, 0.234, 0.891],
    "type": "probability",
    "prediction_intervals": {
        "lower": [0.623, 0.134, 0.791],
        "upper": [0.823, 0.334, 0.991],
        "alpha": 0.05
    }
}
```

### Diagnostics
```
POST /glm/{model_type}/diagnostics
Content-Type: application/json

{
    "X": [[1.2, 3.4], [5.6, 7.8], ...],
    "y": [0, 1, 0, ...],
    "coefficients": [0.523, -0.234, 0.891],
    "intercept": 1.234
}
```

#### Response:
```json
{
    "model_type": "linear",
    "residuals": {
        "raw": [-0.12, 0.34, -0.56, ...],
        "standardized": [-0.45, 1.23, -2.01, ...],
        "deviance": [-0.43, 1.21, -1.98, ...]
    },
    "goodness_of_fit": {
        "r_squared": 0.82,
        "adjusted_r_squared": 0.79,
        "deviance": 123.45,
        "pearson_chi_squared": 98.76
    },
    "influence": {
        "leverage": [0.12, 0.08, 0.15, ...],
        "cooks_distance": [0.02, 0.001, 0.08, ...]
    },
    "tests": {
        "durbin_watson": 1.98,
        "breusch_pagan_p_value": 0.23,
        "shapiro_wilk_p_value": 0.67
    }
}
```

## Data Requirements

### Linear Regression
- Continuous response variable
- No specific constraints on features

### Logistic/Probit Regression
- Binary response variable (0 or 1 only)
- Features can be continuous or categorical (encoded numerically)

### Poisson Regression
- Non-negative integer response variable (counts)
- Features can be continuous or categorical

## Regularization

### L2 (Ridge) Regularization
```json
{
    "regularization": 0.1,
    "regularization_type": "l2"
}
```
- Shrinks coefficients towards zero
- Handles multicollinearity well
- Keeps all features in the model

### L1 (Lasso) Regularization
```json
{
    "regularization": 0.1,
    "regularization_type": "l1"
}
```
- Can set some coefficients exactly to zero
- Performs feature selection
- May be unstable with correlated features

### Elastic Net
```json
{
    "regularization": 0.1,
    "regularization_type": "elastic"
}
```
- Combines L1 and L2 penalties
- Balances feature selection and stability

## Building

### Local Build
```bash
mkdir build
cd build
cmake ..
make
```

### Docker Build
```bash
docker build -t boom/glm-service .
```

## Running

### Local
```bash
./glm_service 0.0.0.0 8085
```

### Docker
```bash
docker run -p 8085:8085 boom/glm-service
```

## Testing Examples

### Linear Regression
```bash
# Fit a linear model
curl -X POST http://localhost:8085/glm/linear/fit \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    "y": [2.1, 3.9, 6.1, 8.0, 9.9],
    "standardize": true
  }'

# Make predictions
curl -X POST http://localhost:8085/glm/linear/predict \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[6, 7], [7, 8]],
    "coefficients": [1.98, 0.01],
    "intercept": 0.1,
    "prediction_intervals": true
  }'
```

### Logistic Regression
```bash
# Fit a logistic model with regularization
curl -X POST http://localhost:8085/glm/logistic/fit \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5]],
    "y": [0, 0, 0, 1, 1, 1],
    "regularization": 0.1,
    "regularization_type": "l2"
  }'

# Get probability predictions
curl -X POST http://localhost:8085/glm/logistic/predict \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[3.5, 3.5], [2, 2]],
    "coefficients": [0.5, 0.3],
    "intercept": -2.1,
    "type": "probability"
  }'
```

### Model Diagnostics
```bash
# Get comprehensive diagnostics
curl -X POST http://localhost:8085/glm/linear/diagnostics \
  -H "Content-Type: application/json" \
  -d '{
    "X": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    "y": [2.1, 3.9, 6.1, 8.0, 9.9],
    "coefficients": [1.98, 0.01],
    "intercept": 0.1
  }'
```

## Performance Considerations

- Uses Eigen for efficient matrix operations
- IRLS (Iteratively Reweighted Least Squares) for GLM fitting
- Supports datasets with up to 100,000 observations
- Response times typically < 500ms for standard operations
- Memory usage scales with O(n*p) for n observations and p features

## Error Handling

- Invalid model type: Returns 400 with error message
- Incompatible data (e.g., non-binary for logistic): Returns 400 with details
- Numerical instability: Returns partial results with convergence flag
- Missing required parameters: Returns 400 with specific field errors

## Best Practices

1. **Feature Scaling**: Use `standardize: true` when features have different scales
2. **Regularization**: Start with small values (0.01-0.1) and tune using cross-validation
3. **Diagnostics**: Always check residual plots and influence measures
4. **Sample Size**: Ensure at least 10-20 observations per feature
5. **Multicollinearity**: Use regularization or remove highly correlated features