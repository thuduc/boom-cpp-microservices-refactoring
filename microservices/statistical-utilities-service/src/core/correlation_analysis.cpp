#include "correlation_analysis.hpp"
#include "descriptive_stats.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

CorrelationResult CorrelationAnalysis::pearson(const std::vector<double>& x,
                                              const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    
    if (x.size() < 3) {
        throw std::invalid_argument("Need at least 3 data points");
    }
    
    size_t n = x.size();
    double mean_x = DescriptiveStats::mean(x);
    double mean_y = DescriptiveStats::mean(y);
    
    double cov = 0.0;
    double var_x = 0.0;
    double var_y = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    CorrelationResult result;
    result.correlation = cov / std::sqrt(var_x * var_y);
    
    // Test statistic for significance
    double t = result.correlation * std::sqrt((n - 2) / (1 - result.correlation * result.correlation));
    double df = n - 2;
    
    // Simplified p-value calculation
    result.p_value = 2 * (1 - std::abs(t) / std::sqrt(df + t * t));
    
    // Confidence interval using Fisher's z-transformation
    double z = fisherZ(result.correlation);
    double se_z = 1.0 / std::sqrt(n - 3);
    double z_critical = 1.96;  // 95% confidence
    
    double z_lower = z - z_critical * se_z;
    double z_upper = z + z_critical * se_z;
    
    result.confidence_interval = {
        inverseFisherZ(z_lower),
        inverseFisherZ(z_upper)
    };
    
    return result;
}

CorrelationResult CorrelationAnalysis::spearman(const std::vector<double>& x,
                                               const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    
    // Rank the data
    std::vector<double> rank_x = rankData(x);
    std::vector<double> rank_y = rankData(y);
    
    // Calculate Pearson correlation on ranks
    return pearson(rank_x, rank_y);
}

CorrelationResult CorrelationAnalysis::kendall(const std::vector<double>& x,
                                              const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    
    size_t n = x.size();
    int concordant = 0;
    int discordant = 0;
    
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            
            if (dx * dy > 0) {
                concordant++;
            } else if (dx * dy < 0) {
                discordant++;
            }
            // Ties are ignored
        }
    }
    
    CorrelationResult result;
    double total = n * (n - 1) / 2.0;
    result.correlation = (concordant - discordant) / total;
    
    // Approximate variance for significance test
    double var = 2 * (2 * n + 5) / (9.0 * n * (n - 1));
    double z = result.correlation / std::sqrt(var);
    
    // Normal approximation for p-value
    result.p_value = 2 * (1 - std::abs(z) / std::sqrt(1 + z * z));
    
    // Confidence interval
    double se = std::sqrt(var);
    result.confidence_interval = {
        result.correlation - 1.96 * se,
        result.correlation + 1.96 * se
    };
    
    return result;
}

double CorrelationAnalysis::partialCorrelation(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              const std::vector<double>& z) {
    // Partial correlation between x and y controlling for z
    auto r_xy = pearson(x, y).correlation;
    auto r_xz = pearson(x, z).correlation;
    auto r_yz = pearson(y, z).correlation;
    
    double numerator = r_xy - r_xz * r_yz;
    double denominator = std::sqrt((1 - r_xz * r_xz) * (1 - r_yz * r_yz));
    
    return numerator / denominator;
}

std::vector<std::vector<double>> CorrelationAnalysis::correlationMatrix(
    const std::vector<std::vector<double>>& data) {
    
    size_t n_vars = data.size();
    std::vector<std::vector<double>> matrix(n_vars, std::vector<double>(n_vars));
    
    for (size_t i = 0; i < n_vars; ++i) {
        for (size_t j = i; j < n_vars; ++j) {
            if (i == j) {
                matrix[i][j] = 1.0;
            } else {
                double corr = pearson(data[i], data[j]).correlation;
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
    }
    
    return matrix;
}

std::vector<double> CorrelationAnalysis::rankData(const std::vector<double>& data) {
    size_t n = data.size();
    
    // Create pairs of (value, index)
    std::vector<std::pair<double, size_t>> indexed_data;
    for (size_t i = 0; i < n; ++i) {
        indexed_data.push_back({data[i], i});
    }
    
    // Sort by value
    std::sort(indexed_data.begin(), indexed_data.end());
    
    // Assign ranks
    std::vector<double> ranks(n);
    
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        double sum_ranks = 0.0;
        
        // Handle ties
        while (j < n && indexed_data[j].first == indexed_data[i].first) {
            sum_ranks += j + 1;  // Ranks start at 1
            j++;
        }
        
        double avg_rank = sum_ranks / (j - i);
        
        while (i < j) {
            ranks[indexed_data[i].second] = avg_rank;
            i++;
        }
    }
    
    return ranks;
}

double CorrelationAnalysis::fisherZ(double r) {
    return 0.5 * std::log((1 + r) / (1 - r));
}

double CorrelationAnalysis::inverseFisherZ(double z) {
    double e2z = std::exp(2 * z);
    return (e2z - 1) / (e2z + 1);
}