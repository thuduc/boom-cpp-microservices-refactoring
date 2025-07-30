# Mixture Models Service

A microservice for fitting and using mixture models including Gaussian Mixture Models, Dirichlet Process Mixtures, and Hierarchical Dirichlet Process models.

## Features

- **Models Supported:**
  - Gaussian Mixture Models (GMM)
  - Dirichlet Process (DP) Mixtures
  - Hierarchical Dirichlet Process (HDP)

- **Inference Methods:**
  - Expectation-Maximization (EM)
  - Gibbs Sampling
  - Collapsed Gibbs Sampling
  - Variational Inference

- **Capabilities:**
  - Model fitting with automatic component selection
  - Hard and soft clustering predictions
  - Density estimation
  - Model sampling
  - Model selection with BIC/AIC
  - Clustering evaluation metrics

## API Endpoints

### Health Check
```http
GET /health
```

### Fit Mixture Model
```http
POST /mixture/{model_type}/fit
```
Model types: `gaussian_mixture`, `dirichlet_process`, `hierarchical_dp`

Request body:
```json
{
  "data": [[1.0, 2.0], [3.0, 4.0], ...],
  "n_components": 3,
  "method": "em",
  "n_iterations": 100,
  "covariance_type": "full"  // For GMM
}
```

### Predict/Cluster
```http
POST /mixture/{model_type}/predict
```

Request body:
```json
{
  "data": [[1.0, 2.0], [3.0, 4.0], ...],
  "parameters": {...},  // Model parameters from fitting
  "type": "hard"  // or "soft" or "density"
}
```

### Sample from Model
```http
POST /mixture/{model_type}/sample
```

Request body:
```json
{
  "parameters": {...},  // Model parameters
  "n_samples": 100,
  "return_components": true
}
```

### Model Selection
```http
POST /mixture/select
```

Request body:
```json
{
  "data": [[1.0, 2.0], [3.0, 4.0], ...],
  "models": ["gaussian_mixture", "dirichlet_process"],
  "criterion": "bic",
  "max_components": 10
}
```

### Clustering Metrics
```http
POST /mixture/metrics
```

Request body:
```json
{
  "data": [[1.0, 2.0], [3.0, 4.0], ...],
  "labels": [0, 1, 0, ...],
  "true_labels": [0, 1, 0, ...]  // Optional for supervised metrics
}
```

## Building and Running

### Local Build
```bash
mkdir build
cd build
cmake ..
make
./mixture_models_service 0.0.0.0 8087
```

### Docker
```bash
docker build -t mixture-models-service .
docker run -p 8087:8087 mixture-models-service
```

## Examples

### Fit a Gaussian Mixture Model
```bash
curl -X POST http://localhost:8087/mixture/gaussian_mixture/fit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]],
    "n_components": 2,
    "method": "em"
  }'
```

### Make Predictions
```bash
curl -X POST http://localhost:8087/mixture/gaussian_mixture/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1.5, 2.5], [4.5, 5.5]],
    "parameters": {
      "weights": [0.5, 0.5],
      "means": [[1.05, 2.05], [5.05, 6.05]],
      "covariances": [...]
    },
    "type": "hard"
  }'
```

## Dependencies

- Boost.Beast (HTTP server)
- Eigen3 (Linear algebra)
- nlohmann/json (JSON parsing)
- C++17 compiler