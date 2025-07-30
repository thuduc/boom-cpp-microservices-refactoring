#include "time_series_utils.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

std::vector<double> TimeSeriesUtils::jsonToVector(const json& data) {
    if (!data.is_array()) {
        throw std::runtime_error("Expected array for time series data");
    }
    
    std::vector<double> result;
    result.reserve(data.size());
    
    for (const auto& val : data) {
        if (!val.is_number()) {
            throw std::runtime_error("Non-numeric value in time series");
        }
        result.push_back(val.get<double>());
    }
    
    return result;
}

json TimeSeriesUtils::vectorToJson(const std::vector<double>& vec) {
    return json(vec);
}

double TimeSeriesUtils::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double TimeSeriesUtils::variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    
    double m = mean(data);
    double sum_sq = 0.0;
    
    for (double x : data) {
        sum_sq += (x - m) * (x - m);
    }
    
    return sum_sq / (data.size() - 1);
}

double TimeSeriesUtils::skewness(const std::vector<double>& data) {
    if (data.size() < 3) return 0.0;
    
    double m = mean(data);
    double s = std::sqrt(variance(data));
    if (s == 0) return 0.0;
    
    double sum_cubed = 0.0;
    for (double x : data) {
        sum_cubed += std::pow((x - m) / s, 3);
    }
    
    return sum_cubed / data.size();
}

double TimeSeriesUtils::kurtosis(const std::vector<double>& data) {
    if (data.size() < 4) return 0.0;
    
    double m = mean(data);
    double s = std::sqrt(variance(data));
    if (s == 0) return 0.0;
    
    double sum_fourth = 0.0;
    for (double x : data) {
        sum_fourth += std::pow((x - m) / s, 4);
    }
    
    return sum_fourth / data.size() - 3.0;  // Excess kurtosis
}

std::vector<double> TimeSeriesUtils::autocorrelation(const std::vector<double>& data, int max_lag) {
    std::vector<double> acf(max_lag + 1);
    
    double m = mean(data);
    double c0 = 0.0;
    
    // Variance (lag 0)
    for (double x : data) {
        c0 += (x - m) * (x - m);
    }
    c0 /= data.size();
    acf[0] = 1.0;
    
    // Autocorrelations
    for (int k = 1; k <= max_lag; ++k) {
        double ck = 0.0;
        for (size_t t = k; t < data.size(); ++t) {
            ck += (data[t] - m) * (data[t-k] - m);
        }
        acf[k] = ck / (data.size() * c0);
    }
    
    return acf;
}

std::vector<double> TimeSeriesUtils::partialAutocorrelation(const std::vector<double>& data, int max_lag) {
    std::vector<double> pacf(max_lag + 1);
    pacf[0] = 1.0;
    
    // Get autocorrelations
    auto acf = autocorrelation(data, max_lag);
    
    // Durbin-Levinson algorithm
    for (int k = 1; k <= max_lag; ++k) {
        Eigen::MatrixXd R(k, k);
        Eigen::VectorXd r(k);
        
        // Build autocorrelation matrix
        for (int i = 0; i < k; ++i) {
            r(i) = acf[i + 1];
            for (int j = 0; j < k; ++j) {
                R(i, j) = acf[std::abs(i - j)];
            }
        }
        
        // Solve for partial autocorrelations
        Eigen::VectorXd phi = R.ldlt().solve(r);
        pacf[k] = phi(k - 1);
    }
    
    return pacf;
}

TestResult TimeSeriesUtils::ljungBoxTest(const std::vector<double>& residuals, int lags) {
    TestResult result;
    
    auto acf = autocorrelation(residuals, lags);
    double n = residuals.size();
    double Q = 0.0;
    
    for (int k = 1; k <= lags; ++k) {
        Q += acf[k] * acf[k] / (n - k);
    }
    Q *= n * (n + 2);
    
    result.statistic = Q;
    result.df = lags;
    
    // Chi-squared p-value approximation
    // For simplicity, using normal approximation for large df
    double z = (Q - lags) / std::sqrt(2.0 * lags);
    result.p_value = 1.0 - 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
    
    return result;
}

std::vector<double> TimeSeriesUtils::movingAverage(const std::vector<double>& data, int window) {
    if (window <= 0 || window > data.size()) return {};
    
    std::vector<double> ma(data.size() - window + 1);
    
    // Initial sum
    double sum = 0.0;
    for (int i = 0; i < window; ++i) {
        sum += data[i];
    }
    ma[0] = sum / window;
    
    // Rolling calculation
    for (size_t i = 1; i < ma.size(); ++i) {
        sum = sum - data[i - 1] + data[i + window - 1];
        ma[i] = sum / window;
    }
    
    return ma;
}

DecompositionResult TimeSeriesUtils::classicalDecomposition(const std::vector<double>& data, 
                                                           int period, 
                                                           bool multiplicative) {
    DecompositionResult result;
    size_t n = data.size();
    
    if (n < 2 * period) {
        throw std::runtime_error("Insufficient data for decomposition");
    }
    
    // Step 1: Compute trend using centered moving average
    result.trend = movingAverage(data, period);
    
    // Center the trend
    if (period % 2 == 0) {
        auto trend2 = movingAverage(result.trend, 2);
        result.trend = trend2;
    }
    
    // Pad trend to match data length
    size_t trend_start = (n - result.trend.size()) / 2;
    std::vector<double> padded_trend(n, 0.0);
    for (size_t i = 0; i < result.trend.size(); ++i) {
        padded_trend[trend_start + i] = result.trend[i];
    }
    result.trend = padded_trend;
    
    // Step 2: Detrend and compute seasonal
    result.seasonal.resize(n);
    std::vector<double> seasonal_avg(period, 0.0);
    std::vector<int> seasonal_count(period, 0);
    
    for (size_t i = trend_start; i < trend_start + result.trend.size(); ++i) {
        if (result.trend[i] != 0) {
            double detrended;
            if (multiplicative) {
                detrended = data[i] / result.trend[i];
            } else {
                detrended = data[i] - result.trend[i];
            }
            
            int season = i % period;
            seasonal_avg[season] += detrended;
            seasonal_count[season]++;
        }
    }
    
    // Average seasonal components
    for (int s = 0; s < period; ++s) {
        if (seasonal_count[s] > 0) {
            seasonal_avg[s] /= seasonal_count[s];
        }
    }
    
    // Normalize seasonal components
    if (multiplicative) {
        double sum = std::accumulate(seasonal_avg.begin(), seasonal_avg.end(), 0.0);
        double target = period;
        for (int s = 0; s < period; ++s) {
            seasonal_avg[s] *= target / sum;
        }
    } else {
        double sum = std::accumulate(seasonal_avg.begin(), seasonal_avg.end(), 0.0);
        for (int s = 0; s < period; ++s) {
            seasonal_avg[s] -= sum / period;
        }
    }
    
    // Fill seasonal component
    for (size_t i = 0; i < n; ++i) {
        result.seasonal[i] = seasonal_avg[i % period];
    }
    
    // Step 3: Compute remainder
    result.remainder.resize(n);
    for (size_t i = 0; i < n; ++i) {
        if (multiplicative && result.trend[i] != 0) {
            result.remainder[i] = data[i] / (result.trend[i] * result.seasonal[i]);
        } else {
            result.remainder[i] = data[i] - result.trend[i] - result.seasonal[i];
        }
    }
    
    return result;
}

DecompositionResult TimeSeriesUtils::stlDecomposition(const std::vector<double>& data, int period) {
    // Simplified STL - in practice, use full STL algorithm
    return classicalDecomposition(data, period, false);
}

std::vector<double> TimeSeriesUtils::boxCoxTransform(const std::vector<double>& data, double lambda) {
    std::vector<double> transformed(data.size());
    
    // Check for positive data
    double min_val = *std::min_element(data.begin(), data.end());
    if (min_val <= 0) {
        throw std::runtime_error("Box-Cox transform requires positive data");
    }
    
    if (std::abs(lambda) < 1e-10) {
        // Log transform
        for (size_t i = 0; i < data.size(); ++i) {
            transformed[i] = std::log(data[i]);
        }
    } else {
        // Power transform
        for (size_t i = 0; i < data.size(); ++i) {
            transformed[i] = (std::pow(data[i], lambda) - 1.0) / lambda;
        }
    }
    
    return transformed;
}

TestResult TimeSeriesUtils::augmentedDickeyFullerTest(const std::vector<double>& data) {
    TestResult result;
    
    // Simplified ADF test - in practice, use full implementation
    size_t n = data.size();
    if (n < 3) {
        result.statistic = 0.0;
        result.p_value = 1.0;
        result.df = 0;
        return result;
    }
    
    // First differences
    std::vector<double> diff(n - 1);
    for (size_t i = 1; i < n; ++i) {
        diff[i-1] = data[i] - data[i-1];
    }
    
    // Simple regression: Δy_t = ρ*y_{t-1} + ε_t
    double sum_y = 0.0, sum_dy = 0.0, sum_y2 = 0.0, sum_y_dy = 0.0;
    
    for (size_t i = 0; i < diff.size(); ++i) {
        sum_y += data[i];
        sum_dy += diff[i];
        sum_y2 += data[i] * data[i];
        sum_y_dy += data[i] * diff[i];
    }
    
    double rho = (diff.size() * sum_y_dy - sum_y * sum_dy) / 
                 (diff.size() * sum_y2 - sum_y * sum_y);
    
    // Test statistic
    result.statistic = rho / 0.1;  // Simplified standard error
    result.df = 1;
    
    // Critical values approximation
    if (result.statistic < -2.86) {
        result.p_value = 0.01;
    } else if (result.statistic < -1.95) {
        result.p_value = 0.05;
    } else {
        result.p_value = 0.10;
    }
    
    return result;
}