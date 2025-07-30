# Statistical Models Service

A microservice providing parameter estimation, prediction, and simulation for common statistical distributions.

## Features

- Support for multiple distributions:
  - Gaussian (Normal)
  - Gamma
  - Beta
  - Poisson
  - Multinomial
- Maximum Likelihood Estimation (MLE)
- Bayesian estimation with conjugate priors
- Probability functions (PDF, CDF)
- Quantile computation
- Random sampling/simulation
- Model comparison (AIC, BIC)

## API Endpoints

### Health Check
```
GET /health
```

### Fit Model
```
POST /models/{model_type}/fit
Content-Type: application/json

{
    "data": [1.2, 2.3, 3.1, ...],
    "method": "mle",  // or "bayesian"
    "prior": {        // Optional, for Bayesian
        "alpha": 1.0,
        "beta": 1.0
    }
}
```

Supported model types:
- `gaussian` or `normal`
- `gamma`
- `beta`
- `poisson`
- `multinomial`

### Predict
```
POST /models/{model_type}/predict
Content-Type: application/json

{
    "parameters": {
        "mean": 0,
        "std_dev": 1
    },
    "type": "mean",  // or "variance", "quantile", "pdf", "cdf"
    "p": 0.95,       // For quantile
    "x": [0, 1, 2]   // For pdf/cdf
}
```

### Simulate
```
POST /models/{model_type}/simulate
Content-Type: application/json

{
    "parameters": {
        "mean": 0,
        "std_dev": 1
    },
    "n_samples": 1000
}
```

### Likelihood
```
POST /models/{model_type}/likelihood
Content-Type: application/json

{
    "parameters": {
        "mean": 0,
        "std_dev": 1
    },
    "data": [1.2, 2.3, 3.1, ...]
}
```

## Response Examples

### Fit Response
```json
{
    "model_type": "gaussian",
    "method": "mle",
    "parameters": {
        "mean": 2.1,
        "std_dev": 0.8
    },
    "log_likelihood": -123.45,
    "convergence": {
        "converged": true,
        "iterations": 1
    },
    "standard_errors": [0.08, 0.056],
    "confidence_intervals": [
        [1.94, 2.26],
        [0.69, 0.91]
    ]
}
```

### Likelihood Response with Model Comparison
```json
{
    "model_type": "gaussian",
    "parameters": {
        "mean": 2.1,
        "std_dev": 0.8
    },
    "log_likelihood": -123.45,
    "likelihood": 1.23e-54,
    "n_observations": 100,
    "aic": 250.9,
    "bic": 256.1
}
```

## Model-Specific Parameters

### Gaussian/Normal
```json
{
    "mean": 0.0,
    "std_dev": 1.0
}
```

### Gamma
```json
{
    "shape": 2.0,
    "scale": 1.0
}
```

### Beta
```json
{
    "alpha": 2.0,
    "beta": 5.0
}
```

### Poisson
```json
{
    "lambda": 3.5
}
```

### Multinomial
```json
{
    "probabilities": [0.2, 0.3, 0.5],
    "trials": 10  // Optional, for simulation
}
```

## Bayesian Priors

### Gaussian - Normal-Inverse-Gamma
```json
{
    "prior": {
        "mean": 0.0,
        "kappa": 1.0,
        "alpha": 1.0,
        "beta": 1.0
    }
}
```

### Gamma - Conjugate Prior
```json
{
    "prior": {
        "shape_prior": 1.0,
        "scale_prior": 1.0,
        "prior_weight": 1.0
    }
}
```

### Beta - Beta Prior
```json
{
    "prior": {
        "alpha": 1.0,
        "beta": 1.0
    }
}
```

### Poisson - Gamma Prior
```json
{
    "prior": {
        "alpha": 1.0,
        "beta": 1.0
    }
}
```

### Multinomial - Dirichlet Prior
```json
{
    "prior": {
        "alpha": [1.0, 1.0, 1.0]
    }
}
```

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
docker build -t boom/statistical-models-service .
```

## Running

### Local
```bash
./statistical_models_service 0.0.0.0 8084
```

### Docker
```bash
docker run -p 8084:8084 boom/statistical-models-service
```

## Testing

```bash
# Health check
curl http://localhost:8084/health

# Fit Gaussian model with MLE
curl -X POST http://localhost:8084/models/gaussian/fit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.2, 2.3, 1.8, 2.1, 2.5, 1.9],
    "method": "mle"
  }'

# Compute PDF
curl -X POST http://localhost:8084/models/gaussian/predict \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {"mean": 0, "std_dev": 1},
    "type": "pdf",
    "x": [-2, -1, 0, 1, 2]
  }'

# Simulate from distribution
curl -X POST http://localhost:8084/models/gamma/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {"shape": 2, "scale": 1},
    "n_samples": 100
  }'
```

## Data Validation

Each model type has specific data requirements:
- **Gaussian**: Any finite real numbers
- **Gamma**: Positive real numbers only
- **Beta**: Values in (0, 1) interval
- **Poisson**: Non-negative integers
- **Multinomial**: Non-negative integers (category indices)

## Performance

- Closed-form solutions for most MLE estimates
- Efficient numerical methods for complex estimates
- Support for datasets up to 1M observations
- Response times typically < 100ms for standard operations

## Error Handling

- Invalid data format: 400 Bad Request with specific error message
- Unknown model type: 400 Bad Request
- Data validation failures: 400 Bad Request with details
- Numerical errors: Appropriate error messages

## Model Selection

Use AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) from the likelihood endpoint to compare models:
- Lower values indicate better model fit
- BIC penalizes model complexity more than AIC
- Consider both criteria along with domain knowledge