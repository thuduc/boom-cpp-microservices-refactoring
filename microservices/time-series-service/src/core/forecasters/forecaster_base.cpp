#include "forecaster_base.hpp"
#include <cmath>
#include <algorithm>

std::vector<double> ForecasterBase::exponentialSmoothing(
    const std::vector<double>& data,
    double alpha,
    int horizon) {
    
    if (data.empty() || horizon <= 0) return {};
    
    // Initialize
    double level = data[0];
    
    // Smooth through the data
    for (size_t t = 1; t < data.size(); ++t) {
        level = alpha * data[t] + (1 - alpha) * level;
    }
    
    // Forecast (constant level)
    return std::vector<double>(horizon, level);
}

std::vector<double> ForecasterBase::holtsMethod(
    const std::vector<double>& data,
    double alpha,
    double beta,
    int horizon) {
    
    if (data.size() < 2 || horizon <= 0) return {};
    
    // Initialize
    double level = data[0];
    double trend = data[1] - data[0];
    
    // Smooth through the data
    for (size_t t = 1; t < data.size(); ++t) {
        double prev_level = level;
        level = alpha * data[t] + (1 - alpha) * (level + trend);
        trend = beta * (level - prev_level) + (1 - beta) * trend;
    }
    
    // Forecast with linear trend
    std::vector<double> forecasts(horizon);
    for (int h = 0; h < horizon; ++h) {
        forecasts[h] = level + (h + 1) * trend;
    }
    
    return forecasts;
}

std::vector<double> ForecasterBase::holtWinters(
    const std::vector<double>& data,
    double alpha,
    double beta,
    double gamma,
    int period,
    int horizon,
    bool multiplicative) {
    
    if (data.size() < 2 * period || horizon <= 0) return {};
    
    // Initialize components
    double level = 0.0;
    double trend = 0.0;
    std::vector<double> seasonal(period);
    
    // Compute initial level and trend
    for (int i = 0; i < period; ++i) {
        level += data[i];
    }
    level /= period;
    
    for (int i = 0; i < period; ++i) {
        trend += (data[period + i] - data[i]);
    }
    trend /= (period * period);
    
    // Compute initial seasonal factors
    if (multiplicative) {
        for (int i = 0; i < period; ++i) {
            double sum = 0.0;
            int count = 0;
            for (size_t j = i; j < data.size(); j += period) {
                sum += data[j] / (level + j * trend);
                count++;
            }
            seasonal[i] = sum / count;
        }
    } else {
        for (int i = 0; i < period; ++i) {
            double sum = 0.0;
            int count = 0;
            for (size_t j = i; j < data.size(); j += period) {
                sum += data[j] - (level + j * trend);
                count++;
            }
            seasonal[i] = sum / count;
        }
    }
    
    // Apply Holt-Winters filter
    for (size_t t = 0; t < data.size(); ++t) {
        int season_idx = t % period;
        double prev_level = level;
        
        if (multiplicative) {
            level = alpha * (data[t] / seasonal[season_idx]) + 
                   (1 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1 - beta) * trend;
            seasonal[season_idx] = gamma * (data[t] / level) + 
                                  (1 - gamma) * seasonal[season_idx];
        } else {
            level = alpha * (data[t] - seasonal[season_idx]) + 
                   (1 - alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1 - beta) * trend;
            seasonal[season_idx] = gamma * (data[t] - level) + 
                                  (1 - gamma) * seasonal[season_idx];
        }
    }
    
    // Generate forecasts
    std::vector<double> forecasts(horizon);
    for (int h = 0; h < horizon; ++h) {
        int season_idx = (data.size() + h) % period;
        
        if (multiplicative) {
            forecasts[h] = (level + (h + 1) * trend) * seasonal[season_idx];
        } else {
            forecasts[h] = level + (h + 1) * trend + seasonal[season_idx];
        }
    }
    
    return forecasts;
}