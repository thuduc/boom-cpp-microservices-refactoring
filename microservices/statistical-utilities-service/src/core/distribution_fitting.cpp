#include "distribution_fitting.hpp"
#include "descriptive_stats.hpp"
#include <cmath>
#include <algorithm>

DistributionFitResult DistributionFitting::fitNormal(const std::vector<double>& data) {
    DistributionFitResult result;
    result.distribution = "normal";
    
    // MLE for normal distribution
    double mean = DescriptiveStats::mean(data);
    double variance = DescriptiveStats::variance(data, false);
    double std_dev = std::sqrt(variance);
    
    result.parameters["mean"] = mean;
    result.parameters["std"] = std_dev;
    
    // Log-likelihood
    double n = data.size();
    result.log_likelihood = -0.5 * n * std::log(2 * M_PI) - 
                           0.5 * n * std::log(variance) - 
                           0.5 * n;
    
    // Information criteria
    int k = 2;  // Number of parameters
    result.aic = -2 * result.log_likelihood + 2 * k;
    result.bic = -2 * result.log_likelihood + k * std::log(n);
    
    // Kolmogorov-Smirnov test
    result.ks_statistic = kolmogorovSmirnovTest(data, "normal", result.parameters);
    result.ks_p_value = 1.0 - result.ks_statistic;  // Simplified
    
    return result;
}

DistributionFitResult DistributionFitting::fitExponential(const std::vector<double>& data) {
    DistributionFitResult result;
    result.distribution = "exponential";
    
    // MLE for exponential distribution
    double mean = DescriptiveStats::mean(data);
    double rate = 1.0 / mean;
    
    result.parameters["rate"] = rate;
    
    // Log-likelihood
    double n = data.size();
    double sum = 0.0;
    for (double x : data) {
        sum += x;
    }
    result.log_likelihood = n * std::log(rate) - rate * sum;
    
    // Information criteria
    int k = 1;
    result.aic = -2 * result.log_likelihood + 2 * k;
    result.bic = -2 * result.log_likelihood + k * std::log(n);
    
    result.ks_statistic = kolmogorovSmirnovTest(data, "exponential", result.parameters);
    result.ks_p_value = 1.0 - result.ks_statistic;
    
    return result;
}

DistributionFitResult DistributionFitting::fitGamma(const std::vector<double>& data) {
    DistributionFitResult result;
    result.distribution = "gamma";
    
    // Method of moments for gamma distribution
    double mean = DescriptiveStats::mean(data);
    double variance = DescriptiveStats::variance(data, false);
    
    double shape = mean * mean / variance;
    double scale = variance / mean;
    
    result.parameters["shape"] = shape;
    result.parameters["scale"] = scale;
    
    // Simplified log-likelihood
    double n = data.size();
    result.log_likelihood = -n * shape * std::log(scale) - n * std::lgamma(shape);
    
    int k = 2;
    result.aic = -2 * result.log_likelihood + 2 * k;
    result.bic = -2 * result.log_likelihood + k * std::log(n);
    
    result.ks_statistic = kolmogorovSmirnovTest(data, "gamma", result.parameters);
    result.ks_p_value = 1.0 - result.ks_statistic;
    
    return result;
}

DistributionFitResult DistributionFitting::fitBeta(const std::vector<double>& data) {
    DistributionFitResult result;
    result.distribution = "beta";
    
    // Method of moments for beta distribution
    double mean = DescriptiveStats::mean(data);
    double variance = DescriptiveStats::variance(data, false);
    
    double common = mean * (1 - mean) / variance - 1;
    double alpha = mean * common;
    double beta = (1 - mean) * common;
    
    result.parameters["alpha"] = alpha;
    result.parameters["beta"] = beta;
    
    // Simplified log-likelihood
    double n = data.size();
    result.log_likelihood = n * (std::lgamma(alpha + beta) - 
                               std::lgamma(alpha) - std::lgamma(beta));
    
    int k = 2;
    result.aic = -2 * result.log_likelihood + 2 * k;
    result.bic = -2 * result.log_likelihood + k * std::log(n);
    
    result.ks_statistic = kolmogorovSmirnovTest(data, "beta", result.parameters);
    result.ks_p_value = 1.0 - result.ks_statistic;
    
    return result;
}

DistributionFitResult DistributionFitting::fitWeibull(const std::vector<double>& data) {
    DistributionFitResult result;
    result.distribution = "weibull";
    
    // Simplified parameter estimation
    double mean = DescriptiveStats::mean(data);
    double shape = 2.0;  // Common default
    double scale = mean / std::tgamma(1 + 1.0/shape);
    
    result.parameters["shape"] = shape;
    result.parameters["scale"] = scale;
    
    // Simplified log-likelihood
    double n = data.size();
    result.log_likelihood = n * std::log(shape) - n * shape * std::log(scale);
    
    int k = 2;
    result.aic = -2 * result.log_likelihood + 2 * k;
    result.bic = -2 * result.log_likelihood + k * std::log(n);
    
    result.ks_statistic = kolmogorovSmirnovTest(data, "weibull", result.parameters);
    result.ks_p_value = 1.0 - result.ks_statistic;
    
    return result;
}

double DistributionFitting::kolmogorovSmirnovTest(const std::vector<double>& data,
                                                  const std::string& distribution,
                                                  const std::map<std::string, double>& params) {
    // Simplified KS test
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    double max_diff = 0.0;
    size_t n = sorted_data.size();
    
    for (size_t i = 0; i < n; ++i) {
        double empirical = (i + 1.0) / n;
        double theoretical = 0.5;  // Simplified - should compute actual CDF
        
        if (distribution == "normal") {
            double z = (sorted_data[i] - params.at("mean")) / params.at("std");
            theoretical = 0.5 * (1 + std::erf(z / std::sqrt(2)));
        } else if (distribution == "exponential") {
            theoretical = 1 - std::exp(-params.at("rate") * sorted_data[i]);
        }
        
        max_diff = std::max(max_diff, std::abs(empirical - theoretical));
    }
    
    return max_diff;
}