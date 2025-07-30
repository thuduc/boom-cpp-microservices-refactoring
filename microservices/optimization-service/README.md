# Optimization Service

A microservice providing various numerical optimization algorithms for finding minima of functions.

## Features

- Multiple optimization algorithms:
  - BFGS (Quasi-Newton method)
  - Nelder-Mead (Simplex method)
  - Powell's conjugate direction method
  - Simulated Annealing
  - Newton's method
- Asynchronous job processing
- Support for gradients and Hessians
- Built-in test functions (Rosenbrock, Sphere, Rastrigin)
- RESTful API with job tracking

## API Endpoints

### Health Check
```
GET /health
```

### BFGS Optimization
```
POST /optimize/bfgs
Content-Type: application/json

{
    "objective": "rosenbrock",
    "initial_point": [0.0, 0.0],
    "max_iterations": 1000,
    "tolerance": 1e-6,
    "use_gradient": true,
    "gradient": "rosenbrock"
}
```

### Nelder-Mead Optimization
```
POST /optimize/nelder-mead
Content-Type: application/json

{
    "objective": "sphere",
    "initial_point": [1.0, 2.0, 3.0],
    "max_iterations": 1000,
    "tolerance": 1e-6,
    "simplex_size": 0.1
}
```

### Powell's Method
```
POST /optimize/powell
Content-Type: application/json

{
    "objective": {
        "coefficients": [1, -2, 1],
        "powers": [[2, 0], [1, 0], [0, 0]]
    },
    "initial_point": [5.0],
    "max_iterations": 1000,
    "tolerance": 1e-6
}
```

### Simulated Annealing
```
POST /optimize/simulated-annealing
Content-Type: application/json

{
    "objective": "rastrigin",
    "initial_point": [5.0, 5.0],
    "max_iterations": 10000,
    "initial_temperature": 10.0,
    "cooling_rate": 0.95,
    "step_size": 0.5,
    "lower_bounds": [-5.12, -5.12],
    "upper_bounds": [5.12, 5.12]
}
```

### Newton's Method
```
POST /optimize/newton
Content-Type: application/json

{
    "objective": "sphere",
    "gradient": "sphere",
    "hessian": "sphere",
    "initial_point": [1.0, 1.0],
    "max_iterations": 100,
    "tolerance": 1e-8
}
```

### Check Job Status
```
GET /optimize/status/{job_id}
```

## Response Format

### Job Creation Response
```json
{
    "job_id": "job-a1b2-c3d4-e5f6-789012345678",
    "status": "running",
    "method": "bfgs",
    "message": "Optimization job started"
}
```

### Job Status Response (Completed)
```json
{
    "job_id": "job-a1b2-c3d4-e5f6-789012345678",
    "status": "completed",
    "method": "bfgs",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:05Z",
    "result": {
        "optimal_point": [1.0, 1.0],
        "optimal_value": 0.0,
        "iterations": 25,
        "converged": true,
        "convergence_reason": "Gradient norm below tolerance",
        "final_gradient": [0.0, 0.0]
    }
}
```

## Objective Function Formats

### 1. Built-in Functions
```json
{
    "objective": "rosenbrock"  // or "sphere", "rastrigin"
}
```

### 2. Polynomial Functions
```json
{
    "objective": {
        "coefficients": [1, -2, 3],
        "powers": [[2, 0], [1, 1], [0, 2]]
    }
}
```
This represents: f(x,y) = x² - 2xy + 3y²

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
docker build -t boom/optimization-service .
```

## Running

### Local
```bash
./optimization_service 0.0.0.0 8082
```

### Docker
```bash
docker run -p 8082:8082 boom/optimization-service
```

## Testing

```bash
# Health check
curl http://localhost:8082/health

# Optimize Rosenbrock function with BFGS
curl -X POST http://localhost:8082/optimize/bfgs \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "rosenbrock",
    "initial_point": [-1.0, 2.0],
    "use_gradient": true,
    "gradient": "rosenbrock"
  }'

# Check job status
curl http://localhost:8082/optimize/status/job-a1b2-c3d4-e5f6-789012345678
```

## Performance Considerations

- Jobs run asynchronously in separate threads
- Large optimization problems may take significant time
- Results are stored in memory (consider adding persistence for production)
- Numerical gradients/Hessians are computed when not provided

## Algorithm Selection Guide

- **BFGS**: Best for smooth functions with available gradients
- **Nelder-Mead**: Good for non-smooth functions, no derivatives needed
- **Powell**: Effective for smooth functions without derivatives
- **Simulated Annealing**: Global optimization, good for multimodal functions
- **Newton**: Fast convergence for smooth functions with Hessian

## Limitations

- Maximum problem dimension: 1000 variables
- Job results stored in memory (cleared after 24 hours)
- Synchronous function evaluation (no parallelization within algorithms)

## Error Handling

- Invalid function format: 400 Bad Request
- Job not found: 404 Not Found
- Optimization failure: Job marked as failed with error message