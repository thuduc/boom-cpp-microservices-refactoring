#include "statistics_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

double StatisticsUtils::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double StatisticsUtils::variance(const std::vector<double>& data) {
    if (data.size() <= 1) return 0.0;
    
    double m = mean(data);
    double sum_sq_diff = 0.0;
    
    for (double x : data) {
        sum_sq_diff += (x - m) * (x - m);
    }
    
    return sum_sq_diff / (data.size() - 1);
}

double StatisticsUtils::standardDeviation(const std::vector<double>& data) {
    return std::sqrt(variance(data));
}

double StatisticsUtils::quantile(const std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    if (p <= 0) return *std::min_element(data.begin(), data.end());
    if (p >= 1) return *std::max_element(data.begin(), data.end());
    
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    double index = p * (sorted.size() - 1);
    int lower = static_cast<int>(std::floor(index));
    int upper = static_cast<int>(std::ceil(index));
    
    if (lower == upper) {
        return sorted[lower];
    }
    
    double weight = index - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}