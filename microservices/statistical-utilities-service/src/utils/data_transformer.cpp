#include "data_transformer.hpp"
#include "../core/descriptive_stats.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

std::vector<double> DataTransformer::normalize(const std::vector<double>& data,
                                              const std::string& method) {
    if (data.empty()) {
        return std::vector<double>();
    }
    
    std::vector<double> normalized;
    normalized.reserve(data.size());
    
    if (method == "minmax") {
        double min_val = *std::min_element(data.begin(), data.end());
        double max_val = *std::max_element(data.begin(), data.end());
        double range = max_val - min_val;
        
        if (range == 0) {
            return std::vector<double>(data.size(), 0.5);
        }
        
        for (double value : data) {
            normalized.push_back((value - min_val) / range);
        }
    } else if (method == "sum") {
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        
        if (sum == 0) {
            return std::vector<double>(data.size(), 1.0 / data.size());
        }
        
        for (double value : data) {
            normalized.push_back(value / sum);
        }
    }
    
    return normalized;
}

std::vector<double> DataTransformer::standardize(const std::vector<double>& data) {
    if (data.size() < 2) {
        return data;
    }
    
    double mean = DescriptiveStats::mean(data);
    double std_dev = DescriptiveStats::standardDeviation(data);
    
    if (std_dev == 0) {
        return std::vector<double>(data.size(), 0.0);
    }
    
    std::vector<double> standardized;
    standardized.reserve(data.size());
    
    for (double value : data) {
        standardized.push_back((value - mean) / std_dev);
    }
    
    return standardized;
}

std::vector<double> DataTransformer::logTransform(const std::vector<double>& data) {
    std::vector<double> transformed;
    transformed.reserve(data.size());
    
    for (double value : data) {
        if (value <= 0) {
            throw std::domain_error("Log transform requires positive values");
        }
        transformed.push_back(std::log(value));
    }
    
    return transformed;
}

std::vector<double> DataTransformer::boxCoxTransform(const std::vector<double>& data,
                                                    double lambda) {
    std::vector<double> transformed;
    transformed.reserve(data.size());
    
    for (double value : data) {
        if (value <= 0) {
            throw std::domain_error("Box-Cox transform requires positive values");
        }
        
        if (std::abs(lambda) < 1e-10) {
            transformed.push_back(std::log(value));
        } else {
            transformed.push_back((std::pow(value, lambda) - 1) / lambda);
        }
    }
    
    return transformed;
}

std::vector<double> DataTransformer::difference(const std::vector<double>& data,
                                               int order) {
    if (order <= 0) {
        return data;
    }
    
    if (order >= data.size()) {
        return std::vector<double>();
    }
    
    std::vector<double> result = data;
    
    for (int i = 0; i < order; ++i) {
        std::vector<double> temp;
        for (size_t j = 1; j < result.size(); ++j) {
            temp.push_back(result[j] - result[j-1]);
        }
        result = temp;
    }
    
    return result;
}

std::vector<std::vector<double>> DataTransformer::createLagFeatures(
    const std::vector<double>& data,
    int max_lag) {
    
    if (max_lag <= 0 || max_lag >= data.size()) {
        throw std::invalid_argument("Invalid lag value");
    }
    
    size_t n = data.size();
    std::vector<std::vector<double>> lag_features(max_lag + 1);
    
    // Original series (lag 0)
    lag_features[0] = std::vector<double>(data.begin() + max_lag, data.end());
    
    // Lagged series
    for (int lag = 1; lag <= max_lag; ++lag) {
        lag_features[lag] = std::vector<double>(
            data.begin() + max_lag - lag,
            data.end() - lag
        );
    }
    
    return lag_features;
}