# Linear Algebra Service

A high-performance microservice providing linear algebra operations powered by Eigen.

## Features

- Matrix operations (multiply, add, subtract, transpose, inverse)
- Matrix decompositions (Cholesky, LU, QR, SVD)
- Linear system solvers
- Eigenvalue computation
- RESTful API with JSON input/output

## API Endpoints

### Health Check
```
GET /health
```

### Matrix Multiplication
```
POST /linalg/multiply
Content-Type: application/json

{
    "matrix_a": [[1, 2], [3, 4]],
    "matrix_b": [[5, 6], [7, 8]]
}
```

### Cholesky Decomposition
```
POST /linalg/decompose/cholesky
Content-Type: application/json

{
    "matrix": [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
}
```

### LU Decomposition
```
POST /linalg/decompose/lu
Content-Type: application/json

{
    "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
}
```

### QR Decomposition
```
POST /linalg/decompose/qr
Content-Type: application/json

{
    "matrix": [[1, 2], [3, 4], [5, 6]]
}
```

### SVD Decomposition
```
POST /linalg/decompose/svd
Content-Type: application/json

{
    "matrix": [[1, 2], [3, 4], [5, 6]]
}
```

### Linear System Solver
```
POST /linalg/solve
Content-Type: application/json

{
    "A": [[3, 2], [1, 2]],
    "b": [7, 4],
    "method": "auto"  // Options: "auto", "cholesky", "lu", "qr"
}
```

### Eigenvalues and Eigenvectors
```
POST /linalg/eigenvalues
Content-Type: application/json

{
    "matrix": [[1, 2], [2, 1]],
    "compute_eigenvectors": true
}
```

## Response Format

All successful responses include:
- `operation`: The operation performed
- `result`: The computed result
- `computation_time_us`: Time taken in microseconds
- Additional metadata specific to each operation

Example response:
```json
{
    "operation": "multiply",
    "result": [[19, 22], [43, 50]],
    "computation_time_us": 245,
    "dimensions": {
        "input_a": [2, 2],
        "input_b": [2, 2],
        "output": [2, 2]
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
docker build -t boom/linalg-service .
```

## Running

### Local
```bash
./linalg_service 0.0.0.0 8081
```

### Docker
```bash
docker run -p 8081:8081 boom/linalg-service
```

## Testing

```bash
# Health check
curl http://localhost:8081/health

# Matrix multiplication
curl -X POST http://localhost:8081/linalg/multiply \
  -H "Content-Type: application/json" \
  -d '{
    "matrix_a": [[1, 2, 3], [4, 5, 6]],
    "matrix_b": [[7, 8], [9, 10], [11, 12]]
  }'

# Solve linear system
curl -X POST http://localhost:8081/linalg/solve \
  -H "Content-Type: application/json" \
  -d '{
    "A": [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]],
    "b": [1, -2, 0]
  }'
```

## Performance

- Leverages Eigen's optimized SIMD implementations
- Thread-safe operation handling
- Supports matrices up to 10MB in JSON payload
- Average response times:
  - Matrix multiplication (100x100): <5ms
  - SVD (100x100): <20ms
  - Eigenvalue computation (100x100): <15ms

## Error Handling

The service validates all inputs and returns appropriate error messages:
- `400 Bad Request`: Invalid input format or incompatible dimensions
- `404 Not Found`: Unknown endpoint
- `500 Internal Server Error`: Computation failures (e.g., singular matrix)

## Security

- Input size validation to prevent DoS
- Numerical stability checks
- No persistent state or data storage

## Matrix Format

Matrices are represented as arrays of arrays in row-major order:
```json
[
    [1, 2, 3],    // First row
    [4, 5, 6],    // Second row
    [7, 8, 9]     // Third row
]
```

Vectors can be represented as simple arrays:
```json
[1, 2, 3, 4, 5]
```

Complex numbers are represented as objects:
```json
{
    "real": 3.14,
    "imag": 2.71
}
```

## Limitations

- Maximum matrix size: 5000x5000 elements
- Request timeout: 30 seconds
- Numerical precision: IEEE 754 double precision

## Dependencies

- Eigen 3.4+ for linear algebra operations
- Boost.Beast for HTTP server
- nlohmann/json for JSON parsing
- C++17 compiler support required