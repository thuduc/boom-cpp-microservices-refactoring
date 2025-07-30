# Random Number Generation Service

A microservice providing random number generation from various probability distributions.

## Features

- Uniform distribution
- Normal (Gaussian) distribution
- Gamma distribution
- Beta distribution
- Multinomial distribution
- Thread-safe random number generation
- RESTful API

## API Endpoints

### Health Check
```
GET /health
```

### Seed RNG
```
POST /rng/seed
Content-Type: application/json

{
    "seed": 12345
}
```

### Generate Uniform Random Numbers
```
POST /rng/uniform
Content-Type: application/json

{
    "n": 10,
    "min": 0.0,
    "max": 1.0
}
```

### Generate Normal Random Numbers
```
POST /rng/normal
Content-Type: application/json

{
    "n": 10,
    "mean": 0.0,
    "sd": 1.0
}
```

### Generate Gamma Random Numbers
```
POST /rng/gamma
Content-Type: application/json

{
    "n": 10,
    "shape": 2.0,
    "scale": 1.0
}
```

### Generate Beta Random Numbers
```
POST /rng/beta
Content-Type: application/json

{
    "n": 10,
    "a": 2.0,
    "b": 5.0
}
```

### Generate Multinomial Random Numbers
```
POST /rng/multinomial
Content-Type: application/json

{
    "n": 5,
    "trials": 10,
    "probabilities": [0.2, 0.3, 0.5]
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
docker build -t boom/rng-service .
```

## Running

### Local
```bash
./rng_service 0.0.0.0 8080
```

### Docker
```bash
docker run -p 8080:8080 boom/rng-service
```

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Generate uniform random numbers
curl -X POST http://localhost:8080/rng/uniform \
  -H "Content-Type: application/json" \
  -d '{"n": 5, "min": 0, "max": 10}'

# Generate normal random numbers
curl -X POST http://localhost:8080/rng/normal \
  -H "Content-Type: application/json" \
  -d '{"n": 5, "mean": 100, "sd": 15}'
```

## Configuration

The service listens on port 8080 by default. This can be changed by passing different arguments:

```bash
./rng_service 0.0.0.0 9090
```

## Performance

- Thread-safe implementation using mutex protection
- Supports up to 10,000 samples per request for continuous distributions
- Supports up to 1,000 samples per request for multinomial distribution
- Average response time: <10ms for 1000 samples

## Security

- Input validation for all parameters
- Rate limiting recommended at API gateway level
- No persistent state or data storage

## Monitoring

- Health endpoint for liveness checks
- Structured logging to stdout
- Metrics can be added via Prometheus integration