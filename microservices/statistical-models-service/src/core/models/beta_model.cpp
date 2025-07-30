#include "beta_model.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

BetaModel::BetaModel() : alpha_(1.0), beta_(1.0), rng_(std::random_device{}()) {}

double BetaModel::variance() const {
    double sum = alpha_ + beta_;
    return (alpha_ * beta_) / (sum * sum * (sum + 1));
}

FitResult BetaModel::fitMLE(const std::vector<double>& data) {
    FitResult result;
    
    if (data.empty()) {
        result.converged = false;
        return result;
    }
    
    // Method of moments for initial estimates
    double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sample_var = 0.0;
    for (double x : data) {
        sample_var += (x - sample_mean) * (x - sample_mean);
    }
    sample_var /= (data.size() - 1);
    
    // Initial estimates using method of moments
    double temp = sample_mean * (1 - sample_mean) / sample_var - 1;
    alpha_ = sample_mean * temp;
    beta_ = (1 - sample_mean) * temp;
    
    // Ensure positive parameters
    alpha_ = std::max(alpha_, 0.1);
    beta_ = std::max(beta_, 0.1);
    
    // Could implement Newton-Raphson here for better estimates
    // For now, using method of moments
    
    result.parameters = {
        {"alpha", alpha_},
        {"beta", beta_}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;
    
    // Approximate standard errors
    double n = data.size();
    double info_aa = n * (trigamma(alpha_) - trigamma(alpha_ + beta_));
    double info_bb = n * (trigamma(beta_) - trigamma(alpha_ + beta_));
    double info_ab = -n * trigamma(alpha_ + beta_);
    
    // Inverse of information matrix
    double det = info_aa * info_bb - info_ab * info_ab;
    double se_alpha = std::sqrt(info_bb / det);
    double se_beta = std::sqrt(info_aa / det);
    
    result.standard_errors = {se_alpha, se_beta};
    
    // 95% confidence intervals
    double z = 1.96;
    result.confidence_intervals = {
        {alpha_ - z * se_alpha, alpha_ + z * se_alpha},
        {beta_ - z * se_beta, beta_ + z * se_beta}
    };
    
    return result;
}

FitResult BetaModel::fitBayesian(const std::vector<double>& data, const json& prior_params) {
    FitResult result;
    
    // Beta-Binomial conjugate prior
    double prior_alpha = prior_params.value("alpha", 1.0);
    double prior_beta = prior_params.value("beta", 1.0);
    
    // Update with data (treating each observation as success/failure)
    double sum_x = std::accumulate(data.begin(), data.end(), 0.0);
    double sum_1_minus_x = 0.0;
    for (double x : data) {
        sum_1_minus_x += (1 - x);
    }
    
    // Posterior parameters
    alpha_ = prior_alpha + sum_x;
    beta_ = prior_beta + sum_1_minus_x;
    
    result.parameters = {
        {"alpha", alpha_},
        {"beta", beta_}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;
    
    return result;
}

void BetaModel::setParameters(const json& params) {
    alpha_ = params.value("alpha", 1.0);
    beta_ = params.value("beta", 1.0);
}

json BetaModel::getParameters() const {
    return {
        {"alpha", alpha_},
        {"beta", beta_}
    };
}

double BetaModel::quantile(double p) const {
    // Use bisection method to find quantile
    if (p <= 0) return 0.0;
    if (p >= 1) return 1.0;
    
    double low = 0.0;
    double high = 1.0;
    double mid;
    
    for (int i = 0; i < 50; ++i) {
        mid = (low + high) / 2.0;
        double cdf_mid = cdf(mid);
        
        if (std::abs(cdf_mid - p) < 1e-6) break;
        
        if (cdf_mid < p) {
            low = mid;
        } else {
            high = mid;
        }
    }
    
    return mid;
}

double BetaModel::pdf(double x) const {
    if (x <= 0 || x >= 1) return 0.0;
    
    double log_pdf = (alpha_ - 1) * std::log(x) + (beta_ - 1) * std::log(1 - x) -
                     (std::lgamma(alpha_) + std::lgamma(beta_) - std::lgamma(alpha_ + beta_));
    
    return std::exp(log_pdf);
}

double BetaModel::cdf(double x) const {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;
    
    // Regularized incomplete beta function
    // Simplified implementation using series expansion
    double sum = 0.0;
    double term = 1.0;
    
    for (int k = 0; k < 100; ++k) {
        if (k > 0) {
            term *= (x * (alpha_ + k - 1)) / (k * (alpha_ + beta_ + k - 1));
        }
        sum += term * std::pow(1 - x, beta_);
        
        if (term < 1e-10) break;
    }
    
    return sum * std::exp(alpha_ * std::log(x) + std::lgamma(alpha_ + beta_) - 
                         std::lgamma(alpha_) - std::lgamma(beta_));
}

double BetaModel::logLikelihood(const std::vector<double>& data) const {
    double log_lik = 0.0;
    double log_const = std::lgamma(alpha_ + beta_) - std::lgamma(alpha_) - std::lgamma(beta_);
    
    for (double x : data) {
        if (x <= 0 || x >= 1) return -std::numeric_limits<double>::infinity();
        
        log_lik += log_const + (alpha_ - 1) * std::log(x) + (beta_ - 1) * std::log(1 - x);
    }
    
    return log_lik;
}

std::vector<double> BetaModel::simulate(int n_samples) const {
    // Use gamma variates to generate beta
    std::gamma_distribution<> gamma_a(alpha_, 1.0);
    std::gamma_distribution<> gamma_b(beta_, 1.0);
    
    std::vector<double> samples(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        double x = gamma_a(rng_);
        double y = gamma_b(rng_);
        samples[i] = x / (x + y);
    }
    
    return samples;
}

double BetaModel::trigamma(double x) {
    // Simple approximation of trigamma function
    if (x <= 0) return 0.0;
    
    double result = 0.0;
    while (x < 10) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    
    // Asymptotic expansion
    result += 1.0 / x + 0.5 / (x * x);
    result += 1.0 / (6.0 * x * x * x);
    
    return result;
}