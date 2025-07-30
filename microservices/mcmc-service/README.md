# MCMC Sampling Service

A microservice providing Markov Chain Monte Carlo (MCMC) sampling algorithms for Bayesian inference and probabilistic modeling.

## Features

- Multiple MCMC algorithms:
  - Metropolis-Hastings with adaptive proposals
  - Slice sampling
  - Adaptive Rejection Metropolis Sampling (ARMS)
  - Gibbs sampling
- Asynchronous chain execution
- Chain diagnostics (R-hat, ESS, autocorrelation, Geweke)
- Support for various target distributions
- RESTful API with chain management

## API Endpoints

### Health Check
```
GET /health
```

### Metropolis-Hastings Sampler
```
POST /mcmc/metropolis-hastings
Content-Type: application/json

{
    "target_distribution": "standard_normal",
    "initial_state": [0.0],
    "num_samples": 10000,
    "burn_in": 1000,
    "thin": 2,
    "adaptation_period": 500,
    "proposal": {
        "type": "normal",
        "scale": 1.0
    }
}
```

### Slice Sampler
```
POST /mcmc/slice
Content-Type: application/json

{
    "target_distribution": {
        "type": "multivariate_normal",
        "mean": [0.0, 0.0],
        "covariance": [[1.0, 0.5], [0.5, 1.0]]
    },
    "initial_state": [1.0, 1.0],
    "num_samples": 5000,
    "burn_in": 500,
    "slice_width": 2.0,
    "max_stepping_out": 20
}
```

### ARMS Sampler
```
POST /mcmc/arms
Content-Type: application/json

{
    "target_distribution": "standard_cauchy",
    "initial_state": [0.0],
    "num_samples": 5000,
    "burn_in": 500,
    "bounds": {
        "lower": [-10.0],
        "upper": [10.0]
    }
}
```

### Gibbs Sampler
```
POST /mcmc/gibbs
Content-Type: application/json

{
    "conditional_distributions": [
        {"type": "normal", "mean": 0.0, "std_dev": 1.0},
        {"type": "normal", "mean": 0.0, "std_dev": 1.0}
    ],
    "initial_state": [0.0, 0.0],
    "num_samples": 5000,
    "burn_in": 500,
    "random_scan": true
}
```

### Get Chain Results
```
GET /mcmc/chain/{chain_id}
```

### Chain Diagnostics
```
POST /mcmc/diagnostics
Content-Type: application/json

{
    "chain_id": "chain-a1b2-c3d4-e5f6-789012345678"
}
```

## Response Format

### Chain Creation Response
```json
{
    "chain_id": "chain-a1b2-c3d4-e5f6-789012345678",
    "status": "running",
    "method": "metropolis-hastings",
    "message": "MCMC chain started"
}
```

### Chain Status Response (Completed)
```json
{
    "chain_id": "chain-a1b2-c3d4-e5f6-789012345678",
    "status": "completed",
    "method": "metropolis-hastings",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:31:00Z",
    "summary": {
        "num_samples": 5000,
        "acceptance_rate": 0.234,
        "effective_sample_size": [2341.5, 2156.7]
    },
    "preview_samples": [[0.1, 0.2], [0.3, 0.4], ...],
    "download_url": "/mcmc/chain/chain-a1b2-c3d4-e5f6-789012345678/download"
}
```

### Diagnostics Response
```json
{
    "chain_id": "chain-a1b2-c3d4-e5f6-789012345678",
    "diagnostics": {
        "mean": [0.02, -0.01],
        "std_dev": [0.98, 1.02],
        "quantiles": {
            "0.025": [-1.92, -2.01],
            "0.5": [0.01, -0.02],
            "0.975": [1.95, 2.03]
        },
        "autocorrelation": [[1.0, 0.1, 0.01], [1.0, 0.12, 0.02]],
        "effective_sample_size": [2341.5, 2156.7],
        "gelman_rubin": 1.001,
        "geweke_z": [0.23, -0.15]
    }
}
```

## Target Distribution Formats

### 1. Built-in Distributions
```json
"target_distribution": "standard_normal"  // or "standard_cauchy"
```

### 2. Parametric Distributions
```json
{
    "target_distribution": {
        "type": "normal",
        "mean": 5.0,
        "std_dev": 2.0
    }
}
```

### 3. Multivariate Normal
```json
{
    "target_distribution": {
        "type": "multivariate_normal",
        "mean": [0.0, 0.0],
        "covariance": [[1.0, 0.5], [0.5, 1.0]]
    }
}
```

### 4. Mixture Distributions
```json
{
    "target_distribution": {
        "type": "mixture",
        "weights": [0.3, 0.7],
        "components": [
            {"type": "normal", "mean": -2.0, "std_dev": 1.0},
            {"type": "normal", "mean": 2.0, "std_dev": 0.5}
        ]
    }
}
```

### 5. Custom Distributions
```json
{
    "target_distribution": {
        "type": "custom",
        "expression": "banana",
        "b": 0.1
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
docker build -t boom/mcmc-service .
```

## Running

### Local
```bash
./mcmc_service 0.0.0.0 8083
```

### Docker
```bash
docker run -p 8083:8083 boom/mcmc-service
```

## Testing

```bash
# Health check
curl http://localhost:8083/health

# Run Metropolis-Hastings sampler
curl -X POST http://localhost:8083/mcmc/metropolis-hastings \
  -H "Content-Type: application/json" \
  -d '{
    "target_distribution": "standard_normal",
    "initial_state": [0.0],
    "num_samples": 1000
  }'

# Check chain status
curl http://localhost:8083/mcmc/chain/chain-a1b2-c3d4-e5f6-789012345678

# Get diagnostics
curl -X POST http://localhost:8083/mcmc/diagnostics \
  -H "Content-Type: application/json" \
  -d '{"chain_id": "chain-a1b2-c3d4-e5f6-789012345678"}'
```

## Performance Considerations

- Chains run asynchronously in separate threads
- Large chains (>100k samples) may consume significant memory
- Adaptive algorithms converge faster for well-behaved targets
- Thinning reduces autocorrelation but increases computation time

## Algorithm Selection Guide

- **Metropolis-Hastings**: General purpose, works for any target
- **Slice Sampling**: Good for multimodal distributions, no tuning required
- **ARMS**: Efficient for log-concave distributions
- **Gibbs**: Efficient when conditional distributions are known

## Diagnostics Interpretation

- **R-hat**: Values close to 1.0 indicate convergence (< 1.1 is good)
- **ESS**: Effective sample size, higher is better
- **Geweke Z**: Values between -2 and 2 indicate convergence
- **Autocorrelation**: Lower values indicate better mixing

## Limitations

- Maximum chain size: 1M samples
- Maximum dimension: 1000 parameters
- Chain results stored in memory (consider persistence for production)
- Simplified ARMS and Gibbs implementations

## Error Handling

- Invalid distribution format: 400 Bad Request
- Chain not found: 404 Not Found
- Sampling failure: Chain marked as failed with error message