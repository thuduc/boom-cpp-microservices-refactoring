# BOOM Microservices

This directory contains the microservices implementation of the BOOM (Bayesian Object Oriented Modeling) library.

## Services

### 1. RNG Service (Port 8080)
Random Number Generation service providing various probability distributions:
- Uniform, Normal, Gamma, Beta, Multinomial distributions
- Thread-safe implementation
- Seedable RNG for reproducibility

### 2. Linear Algebra Service (Port 8081)
High-performance linear algebra operations:
- Matrix operations (multiply, add, inverse, etc.)
- Decompositions (Cholesky, LU, QR, SVD)
- Linear system solvers
- Eigenvalue computation

### 3. Optimization Service (Port 8082)
Numerical optimization algorithms:
- BFGS quasi-Newton method
- Nelder-Mead simplex method
- Powell's conjugate direction method
- Simulated annealing
- Newton's method
- Asynchronous job processing

### 4. MCMC Sampling Service (Port 8083)
Markov Chain Monte Carlo sampling:
- Metropolis-Hastings with adaptive proposals
- Slice sampling
- Adaptive Rejection Metropolis Sampling (ARMS)
- Gibbs sampling
- Chain diagnostics and convergence analysis

## Quick Start

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f
```

### Testing the Services

```bash
# Test RNG Service
curl http://localhost:8080/health
curl -X POST http://localhost:8080/rng/normal \
  -H "Content-Type: application/json" \
  -d '{"n": 5, "mean": 0, "sd": 1}'

# Test Linear Algebra Service
curl http://localhost:8081/health
curl -X POST http://localhost:8081/linalg/multiply \
  -H "Content-Type: application/json" \
  -d '{
    "matrix_a": [[1, 2], [3, 4]],
    "matrix_b": [[5, 6], [7, 8]]
  }'

# Test Optimization Service
curl http://localhost:8082/health
curl -X POST http://localhost:8082/optimize/bfgs \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "rosenbrock",
    "initial_point": [-1.0, 2.0]
  }'

# Test MCMC Service
curl http://localhost:8083/health
curl -X POST http://localhost:8083/mcmc/metropolis-hastings \
  -H "Content-Type: application/json" \
  -d '{
    "target_distribution": "standard_normal",
    "initial_state": [0.0],
    "num_samples": 1000
  }'
```

## Development

### Building Individual Services

```bash
# Build RNG Service
cd rng-service
docker build -t boom/rng-service .

# Build Linear Algebra Service
cd linalg-service
docker build -t boom/linalg-service .
```

### Running Without Docker

Each service can be built and run locally:

```bash
# RNG Service
cd rng-service
mkdir build && cd build
cmake .. && make
./rng_service 0.0.0.0 8080

# Linear Algebra Service
cd linalg-service
mkdir build && cd build
cmake .. && make
./linalg_service 0.0.0.0 8081
```

## Architecture

```
┌─────────────┐     ┌─────────────┐
│   Client    │     │   Client    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│         API Gateway             │
│        (Coming Soon)            │
└────────┬──────────────┬─────────┘
         │              │
         ▼              ▼
┌────────────────┐ ┌────────────────┐
│  RNG Service   │ │ LinAlg Service │
│   Port 8080    │ │   Port 8081    │
└────────────────┘ └────────────────┘
```

## Service Communication

- External clients: REST API with JSON
- Internal communication: Direct HTTP (gRPC planned)
- Service discovery: Currently static (Consul planned)

## Monitoring

Each service provides:
- Health check endpoint at `/health`
- Structured logging to stdout
- Response time metrics in API responses

## Next Steps

1. **API Gateway**: Implement nginx-based gateway for routing
2. **Service Discovery**: Add Consul for dynamic service registration
3. **Additional Services**: 
   - Optimization Service
   - MCMC Sampling Service
   - Statistical Models Service
4. **Monitoring**: Add Prometheus metrics and Grafana dashboards
5. **Client Libraries**: Python, R, Java, and JavaScript SDKs

## Contributing

When adding new services:
1. Follow the existing service structure
2. Include comprehensive README
3. Add to docker-compose.yml
4. Implement health check endpoint
5. Use consistent error response format
6. Add integration tests

## Performance Considerations

- Services are stateless for horizontal scaling
- Consider caching for expensive computations
- Use connection pooling in clients
- Monitor memory usage for large matrix operations

## Security

- Input validation on all endpoints
- Size limits to prevent DoS
- No persistent state or sensitive data storage
- HTTPS recommended for production deployment