# Special Functions Service

A microservice providing mathematical special functions including Bessel functions, Gamma functions, Beta functions, FFT, and more.

## Features

- **Bessel Functions**: J_n(x), Y_n(x), I_n(x), K_n(x)
- **Gamma Functions**: Gamma, LogGamma, Digamma (Psi), Polygamma
- **Beta Functions**: Beta, LogBeta, Regularized Beta
- **FFT**: Fast Fourier Transform (forward and inverse)
- **Hypergeometric Functions**: 1F1, 2F1
- **Elliptic Functions**: Complete and incomplete elliptic integrals

## API Endpoints

### Health Check
```http
GET /health
```

### Bessel Functions
```http
POST /functions/bessel
```
Request body:
```json
{
  "type": "J",  // J, Y, I, or K
  "order": 0,   // Order n
  "x": 5.0      // Argument
}
```

### Gamma Functions
```http
POST /functions/gamma
```
Request body:
```json
{
  "type": "gamma",  // gamma, loggamma, digamma, trigamma
  "x": 5.0
}
```

### Beta Functions
```http
POST /functions/beta
```
Request body:
```json
{
  "type": "beta",  // beta, logbeta, regularized
  "a": 2.0,
  "b": 3.0,
  "x": 0.5  // For regularized beta
}
```

### FFT
```http
POST /functions/fft
```
Request body:
```json
{
  "data": [1.0, 2.0, 3.0, 4.0],
  "inverse": false
}
```

### Polygamma Functions
```http
POST /functions/polygamma
```
Request body:
```json
{
  "n": 1,    // Order
  "x": 5.0
}
```

### Hypergeometric Functions
```http
POST /functions/hypergeometric
```
Request body:
```json
{
  "type": "1F1",  // or "2F1"
  "a": 1.0,
  "b": 2.0,
  "c": 3.0,       // For 2F1
  "x": 0.5
}
```

### Batch Evaluation
```http
POST /functions/batch
```
Request body:
```json
{
  "requests": [
    {"function": "gamma", "params": {"x": 5.0}},
    {"function": "bessel", "params": {"type": "J", "order": 0, "x": 2.0}}
  ]
}
```

## Building and Running

### Local Build
```bash
mkdir build
cd build
cmake ..
make
./special_functions_service 0.0.0.0 8088
```

### Docker
```bash
docker build -t special-functions-service .
docker run -p 8088:8088 special-functions-service
```

## Examples

### Compute Gamma function
```bash
curl -X POST http://localhost:8088/functions/gamma \
  -H "Content-Type: application/json" \
  -d '{"type": "gamma", "x": 5.0}'
```

### Compute Bessel function J_0(2.0)
```bash
curl -X POST http://localhost:8088/functions/bessel \
  -H "Content-Type: application/json" \
  -d '{"type": "J", "order": 0, "x": 2.0}'
```

### Compute FFT
```bash
curl -X POST http://localhost:8088/functions/fft \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 2.0, 3.0, 4.0], "inverse": false}'
```

## Dependencies

- Boost.Beast (HTTP server)
- Eigen3 (Linear algebra)
- C++17 compiler