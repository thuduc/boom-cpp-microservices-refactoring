# Time Series Service

A comprehensive microservice for time series analysis, modeling, and forecasting using state-space models and classical methods.

## Features

### Models
- **ARIMA**: AutoRegressive Integrated Moving Average models
- **Local Level**: Random walk with observation noise
- **Local Linear Trend**: Level and trend components
- **Seasonal**: Seasonal patterns with trend

### Analysis Tools
- **Kalman Filtering**: Forward filtering and backward smoothing
- **Decomposition**: STL and classical decomposition methods
- **Forecasting**: Point forecasts with prediction intervals
- **Diagnostics**: ACF, PACF, Ljung-Box test, ADF test

### Additional Features
- Box-Cox transformations
- Moving averages
- Exponential smoothing methods
- Statistical tests for model validation

## API Endpoints

### Health Check
```
GET /health
```

Returns service status and available features.

### Fit Model
```
POST /timeseries/{model_type}/fit
Content-Type: application/json

{
    "data": [1.2, 2.3, 3.4, ...],
    "timestamps": [1.0, 2.0, 3.0, ...],  // Optional
    "p": 1,  // For ARIMA: AR order
    "d": 0,  // For ARIMA: Differencing order
    "q": 1,  // For ARIMA: MA order
    "period": 12  // For Seasonal models
}
```

#### Model Types:
- `arima` - ARIMA(p,d,q) models
- `local_level` - Local level (random walk plus noise)
- `local_linear_trend` - Local linear trend model
- `seasonal` - Seasonal model with trend

#### Response:
```json
{
    "model_type": "arima",
    "parameters": {
        "p": 1,
        "d": 0,
        "q": 1,
        "ar_coefficients": [0.7],
        "ma_coefficients": [0.3],
        "intercept": 2.5,
        "sigma2": 1.2
    },
    "log_likelihood": -234.56,
    "aic": 475.12,
    "bic": 489.34,
    "converged": true,
    "iterations": 25,
    "residuals": [-0.1, 0.2, ...],
    "state_estimates": [[2.5], [2.6], ...]
}
```

### Forecast
```
POST /timeseries/{model_type}/forecast
Content-Type: application/json

{
    "parameters": {
        "ar_coefficients": [0.7],
        "ma_coefficients": [0.3],
        "intercept": 2.5,
        "sigma2": 1.2
    },
    "horizon": 12,
    "data": [1.2, 2.3, ...],  // Optional historical data
    "confidence": 0.95
}
```

#### Response:
```json
{
    "model_type": "arima",
    "horizon": 12,
    "point_forecast": [3.2, 3.3, 3.4, ...],
    "lower_bound": [2.8, 2.7, 2.6, ...],
    "upper_bound": [3.6, 3.9, 4.2, ...],
    "confidence": 0.95,
    "forecast_errors": [0.2, 0.3, 0.4, ...]
}
```

### Filter/Smooth
```
POST /timeseries/filter
Content-Type: application/json

{
    "data": [1.2, 2.3, 3.4, ...],
    "model_type": "local_level",
    "method": "kalman",
    "smooth": true,  // Optional
    "parameters": {
        "level": 2.5,
        "sigma_obs": 0.5,
        "sigma_level": 0.1
    }
}
```

#### Response:
```json
{
    "method": "kalman",
    "filtered_states": [[2.5], [2.6], ...],
    "filtered_variances": [[0.1], [0.08], ...],
    "predictions": [2.5, 2.6, ...],
    "innovations": [-0.1, 0.2, ...],
    "smoothed_states": [[2.5], [2.61], ...],
    "smoothed_variances": [[0.05], [0.04], ...]
}
```

### Decomposition
```
POST /timeseries/decompose
Content-Type: application/json

{
    "data": [100, 102, 98, ...],
    "method": "stl",  // or "classical"
    "period": 12,
    "multiplicative": false  // For classical method
}
```

#### Response:
```json
{
    "method": "stl",
    "trend": [100, 100.5, 101, ...],
    "seasonal": [-2, 1, 3, ...],
    "remainder": [2, 0.5, -6, ...],
    "period": 12
}
```

### Diagnostics
```
POST /timeseries/diagnostics
Content-Type: application/json

{
    "residuals": [-0.1, 0.2, -0.3, ...],
    "max_lag": 20
}
```

#### Response:
```json
{
    "acf": [1.0, 0.2, -0.1, ...],
    "pacf": [1.0, 0.2, -0.05, ...],
    "ljung_box": {
        "statistic": 15.2,
        "p_value": 0.23,
        "degrees_of_freedom": 20
    },
    "mean": 0.02,
    "variance": 1.05,
    "skewness": -0.1,
    "kurtosis": 0.3
}
```

## Model Details

### ARIMA(p,d,q)
- **p**: Number of autoregressive terms
- **d**: Degree of differencing
- **q**: Number of moving average terms
- Suitable for: Stationary or trend-stationary series
- State space form for efficient computation

### Local Level Model
- State equation: `level[t] = level[t-1] + η[t]`
- Observation equation: `y[t] = level[t] + ε[t]`
- Parameters: `sigma_obs` (observation noise), `sigma_level` (level noise)
- Suitable for: Series with slowly changing level

### Local Linear Trend Model
- State: Level and trend components
- Level equation: `level[t] = level[t-1] + trend[t-1] + η₁[t]`
- Trend equation: `trend[t] = trend[t-1] + η₂[t]`
- Suitable for: Series with changing trend

### Seasonal Model
- Components: Level, trend, and seasonal
- Handles additive and multiplicative seasonality
- Period must be specified (e.g., 12 for monthly data)
- Suitable for: Series with regular seasonal patterns

## Filtering Methods

### Kalman Filter
- Optimal linear filter for state space models
- Forward filtering for real-time estimates
- Backward smoothing for historical analysis
- Handles missing observations

### Classical Filters
- Moving averages
- Exponential smoothing
- Holt's linear trend method
- Holt-Winters seasonal method

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
docker build -t boom/time-series-service .
```

## Running

### Local
```bash
./time_series_service 0.0.0.0 8086
```

### Docker
```bash
docker run -p 8086:8086 boom/time-series-service
```

## Testing Examples

### Fit ARIMA Model
```bash
curl -X POST http://localhost:8086/timeseries/arima/fit \
  -H "Content-Type: application/json" \
  -d '{
    "data": [10.2, 10.5, 10.1, 10.8, 11.2, 10.9, 11.5, 11.8, 11.3, 12.1],
    "p": 1,
    "d": 0,
    "q": 1
  }'
```

### Generate Forecast
```bash
curl -X POST http://localhost:8086/timeseries/arima/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
        "ar_coefficients": [0.7],
        "ma_coefficients": [0.3],
        "intercept": 10.5,
        "sigma2": 0.5
    },
    "horizon": 6,
    "confidence": 0.95
  }'
```

### Seasonal Decomposition
```bash
curl -X POST http://localhost:8086/timeseries/decompose \
  -H "Content-Type: application/json" \
  -d '{
    "data": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
             115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140],
    "method": "classical",
    "period": 12
  }'
```

### Kalman Filtering
```bash
curl -X POST http://localhost:8086/timeseries/filter \
  -H "Content-Type: application/json" \
  -d '{
    "data": [10.2, 10.5, 10.1, 10.8, 11.2],
    "model_type": "local_level",
    "method": "kalman",
    "smooth": true,
    "parameters": {
        "sigma_obs": 0.5,
        "sigma_level": 0.1
    }
  }'
```

## Performance Considerations

- State space formulation enables O(n) filtering
- Efficient matrix operations using Eigen
- Handles series up to 100,000 observations
- Response times typically < 500ms for standard operations

## Best Practices

1. **Model Selection**:
   - Use diagnostics to check residuals
   - Compare AIC/BIC across models
   - Consider seasonal patterns in the data

2. **Forecasting**:
   - Check model stability before forecasting
   - Use appropriate confidence levels
   - Consider forecast horizon limitations

3. **Data Preprocessing**:
   - Handle missing values appropriately
   - Consider transformations for non-stationary series
   - Check for outliers and structural breaks

## Error Handling

- Invalid model parameters: Returns 400 with details
- Numerical instability: Returns partial results with warnings
- Missing data: Handled gracefully in Kalman filter
- Non-convergence: Returns results with convergence flag