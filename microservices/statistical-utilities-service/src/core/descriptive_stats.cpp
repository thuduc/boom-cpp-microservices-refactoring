#include "descriptive_stats.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

double DescriptiveStats::mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute mean of empty data");
    }
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double DescriptiveStats::median(std::vector<double> data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute median of empty data");
    }
    
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    
    if (n % 2 == 0) {
        return (data[n/2 - 1] + data[n/2]) / 2.0;
    } else {
        return data[n/2];
    }
}

std::vector<double> DescriptiveStats::mode(const std::vector<double>& data) {
    if (data.empty()) {
        return std::vector<double>();
    }
    
    std::map<double, int> frequency;
    for (double value : data) {
        frequency[value]++;
    }
    
    int max_freq = 0;
    for (const auto& pair : frequency) {
        max_freq = std::max(max_freq, pair.second);
    }
    
    std::vector<double> modes;
    for (const auto& pair : frequency) {
        if (pair.second == max_freq) {
            modes.push_back(pair.first);
        }
    }
    
    return modes;
}

double DescriptiveStats::variance(const std::vector<double>& data, bool sample) {
    if (data.size() < 2) {
        throw std::invalid_argument("Need at least 2 data points for variance");
    }
    
    double m = mean(data);
    double sum_sq_diff = 0.0;
    
    for (double value : data) {
        double diff = value - m;
        sum_sq_diff += diff * diff;
    }
    
    return sum_sq_diff / (sample ? data.size() - 1 : data.size());
}

double DescriptiveStats::standardDeviation(const std::vector<double>& data, bool sample) {
    return std::sqrt(variance(data, sample));
}

double DescriptiveStats::range(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute range of empty data");
    }
    
    auto minmax = std::minmax_element(data.begin(), data.end());
    return *minmax.second - *minmax.first;
}

double DescriptiveStats::interquartileRange(const std::vector<double>& data) {
    std::vector<double> probs = {0.25, 0.75};
    auto quartiles = quantiles(data, probs);
    return quartiles[1] - quartiles[0];
}

double DescriptiveStats::skewness(const std::vector<double>& data) {
    if (data.size() < 3) {
        throw std::invalid_argument("Need at least 3 data points for skewness");
    }
    
    double m = mean(data);
    double s = standardDeviation(data);
    double n = data.size();
    
    double sum_cubed = 0.0;
    for (double value : data) {
        double z = (value - m) / s;
        sum_cubed += z * z * z;
    }
    
    // Fisher-Pearson standardized moment coefficient
    return (n / ((n - 1) * (n - 2))) * sum_cubed;
}

double DescriptiveStats::kurtosis(const std::vector<double>& data) {
    if (data.size() < 4) {
        throw std::invalid_argument("Need at least 4 data points for kurtosis");
    }
    
    double m = mean(data);
    double s = standardDeviation(data);
    double n = data.size();
    
    double sum_fourth = 0.0;
    for (double value : data) {
        double z = (value - m) / s;
        sum_fourth += z * z * z * z;
    }
    
    // Excess kurtosis (subtract 3 for normal distribution)
    double g2 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum_fourth;
    g2 -= 3 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));
    
    return g2;
}

double DescriptiveStats::min(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute min of empty data");
    }
    
    return *std::min_element(data.begin(), data.end());
}

double DescriptiveStats::max(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute max of empty data");
    }
    
    return *std::max_element(data.begin(), data.end());
}

std::vector<double> DescriptiveStats::quantiles(const std::vector<double>& data,
                                               const std::vector<double>& probs) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute quantiles of empty data");
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    std::vector<double> result;
    size_t n = sorted_data.size();
    
    for (double p : probs) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Probability must be in [0, 1]");
        }
        
        double index = p * (n - 1);
        size_t lower = std::floor(index);
        size_t upper = std::ceil(index);
        
        if (lower == upper) {
            result.push_back(sorted_data[lower]);
        } else {
            double weight = index - lower;
            result.push_back((1 - weight) * sorted_data[lower] + 
                           weight * sorted_data[upper]);
        }
    }
    
    return result;
}

std::vector<double> DescriptiveStats::ecdf(const std::vector<double>& data,
                                          const std::vector<double>& eval_points) {
    if (data.empty()) {
        return std::vector<double>(eval_points.size(), 0.0);
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    std::vector<double> result;
    size_t n = sorted_data.size();
    
    for (double x : eval_points) {
        auto it = std::upper_bound(sorted_data.begin(), sorted_data.end(), x);
        size_t count = std::distance(sorted_data.begin(), it);
        result.push_back(static_cast<double>(count) / n);
    }
    
    return result;
}

Spline DescriptiveStats::fitSpline(const std::vector<double>& x,
                                  const std::vector<double>& y,
                                  int degree,
                                  double smoothing) {
    // Simplified cubic spline implementation
    Spline spline;
    spline.knots = x;
    
    // For now, just store coefficients as y values
    // A full implementation would compute spline coefficients
    spline.coefficients = y;
    
    return spline;
}

double DescriptiveStats::moment(const std::vector<double>& data, int order, double center) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute moment of empty data");
    }
    
    double sum = 0.0;
    for (double value : data) {
        sum += std::pow(value - center, order);
    }
    
    return sum / data.size();
}

// Spline implementation
double Spline::evaluate(double x) const {
    if (knots.empty()) return 0.0;
    
    // Find the interval containing x
    auto it = std::lower_bound(knots.begin(), knots.end(), x);
    
    if (it == knots.begin()) {
        return coefficients.front();
    } else if (it == knots.end()) {
        return coefficients.back();
    }
    
    // Linear interpolation for simplicity
    size_t i = std::distance(knots.begin(), it) - 1;
    double t = (x - knots[i]) / (knots[i + 1] - knots[i]);
    return (1 - t) * coefficients[i] + t * coefficients[i + 1];
}

std::vector<double> Spline::getKnots() const {
    return knots;
}