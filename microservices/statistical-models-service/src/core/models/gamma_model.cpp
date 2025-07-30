#include "gamma_model.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

GammaModel::GammaModel() : shape_(1.0), scale_(1.0), rng_(std::random_device{}()) {}

double GammaModel::digamma(double x) {
    // Simple approximation of digamma function
    if (x <= 0) return 0.0;
    
    double result = 0.0;
    while (x < 10) {
        result -= 1.0 / x;
        x += 1.0;
    }
    
    // Asymptotic expansion
    result += std::log(x) - 0.5 / x;
    double x2 = x * x;
    result -= 1.0 / (12.0 * x2);
    
    return result;
}

double GammaModel::trigamma(double x) {
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

FitResult GammaModel::fitMLE(const std::vector<double>& data) {
    FitResult result;
    
    if (data.empty()) {
        result.converged = false;
        return result;
    }
    
    // Method of moments for initial values
    double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sample_var = 0.0;
    for (double x : data) {
        sample_var += (x - sample_mean) * (x - sample_mean);
    }
    sample_var /= (data.size() - 1);
    
    // Initial estimates
    shape_ = sample_mean * sample_mean / sample_var;
    scale_ = sample_var / sample_mean;
    
    // Newton-Raphson for shape parameter
    double log_mean = 0.0;
    for (double x : data) {
        log_mean += std::log(x);
    }
    log_mean /= data.size();
    
    double s = std::log(sample_mean) - log_mean;
    
    // Iterate to find shape
    const int max_iter = 20;
    const double tol = 1e-6;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        double f = std::log(shape_) - digamma(shape_) - s;
        double f_prime = 1.0 / shape_ - trigamma(shape_);
        
        double shape_new = shape_ - f / f_prime;
        
        if (std::abs(shape_new - shape_) < tol) {
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }
        
        shape_ = shape_new;
    }
    
    // Update scale
    scale_ = sample_mean / shape_;
    
    result.parameters = {
        {"shape", shape_},
        {"scale", scale_}
    };
    
    result.log_likelihood = logLikelihood(data);
    
    // Approximate standard errors
    double n = data.size();
    double se_shape = shape_ / std::sqrt(n * trigamma(shape_));
    double se_scale = scale_ / std::sqrt(n);
    
    result.standard_errors = {se_shape, se_scale};
    
    // 95% confidence intervals
    double z = 1.96;
    result.confidence_intervals = {
        {shape_ - z * se_shape, shape_ + z * se_shape},
        {scale_ - z * se_scale, scale_ + z * se_scale}
    };
    
    return result;
}

FitResult GammaModel::fitBayesian(const std::vector<double>& data, const json& prior_params) {
    FitResult result;
    
    // Use conjugate prior (not exact for gamma, but approximate)
    // Prior on shape: log-normal
    // Prior on rate (1/scale): gamma
    
    // For simplicity, use method of moments with pseudo-observations
    double prior_shape = prior_params.value("shape_prior", 1.0);
    double prior_scale = prior_params.value("scale_prior", 1.0);
    double prior_weight = prior_params.value("prior_weight", 1.0);
    
    // Combine prior and data
    double n = data.size();
    double data_mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    
    // Weighted average
    double post_mean = (prior_weight * prior_shape * prior_scale + n * data_mean) / 
                      (prior_weight + n);
    
    // Approximate posterior
    shape_ = prior_shape;  // Keep prior shape as approximation
    scale_ = post_mean / shape_;
    
    result.parameters = {
        {"shape", shape_},
        {"scale", scale_}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;
    
    return result;
}

void GammaModel::setParameters(const json& params) {
    shape_ = params.value("shape", 1.0);
    scale_ = params.value("scale", 1.0);
}

json GammaModel::getParameters() const {
    return {
        {"shape", shape_},
        {"scale", scale_}
    };
}

double GammaModel::quantile(double p) const {
    // Use Newton's method to find quantile
    if (p <= 0) return 0.0;
    if (p >= 1) return std::numeric_limits<double>::infinity();
    
    // Initial guess using Wilson-Hilferty transformation
    double z = quantile_normal(p);
    double w = 2.0 / (9.0 * shape_);
    double x = shape_ * scale_ * std::pow(1.0 - w + z * std::sqrt(w), 3);
    
    // Refine with Newton's method
    for (int i = 0; i < 10; ++i) {
        double cdf_x = cdf(x);
        double pdf_x = pdf(x);
        
        if (pdf_x == 0) break;
        
        double x_new = x - (cdf_x - p) / pdf_x;
        
        if (std::abs(x_new - x) < 1e-6) break;
        
        x = x_new;
    }
    
    return x;
}

double GammaModel::pdf(double x) const {
    if (x <= 0) return 0.0;
    
    double log_pdf = (shape_ - 1) * std::log(x) - x / scale_ - 
                     shape_ * std::log(scale_) - std::lgamma(shape_);
    
    return std::exp(log_pdf);
}

double GammaModel::cdf(double x) const {
    if (x <= 0) return 0.0;
    
    // Use incomplete gamma function
    // Simplified implementation
    double sum = 0.0;
    double term = 1.0;
    double a = shape_;
    double z = x / scale_;
    
    for (int k = 0; k < 100; ++k) {
        if (k > 0) {
            term *= z / (a + k - 1);
        }
        sum += term;
        
        if (term < 1e-10 * sum) break;
    }
    
    return sum * std::exp(-z + a * std::log(z) - std::lgamma(a));
}

double GammaModel::logLikelihood(const std::vector<double>& data) const {
    double log_lik = 0.0;
    double log_const = -shape_ * std::log(scale_) - std::lgamma(shape_);
    
    for (double x : data) {
        if (x <= 0) return -std::numeric_limits<double>::infinity();
        
        log_lik += log_const + (shape_ - 1) * std::log(x) - x / scale_;
    }
    
    return log_lik;
}

std::vector<double> GammaModel::simulate(int n_samples) const {
    std::gamma_distribution<> dist(shape_, scale_);
    std::vector<double> samples(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        samples[i] = dist(rng_);
    }
    
    return samples;
}

double GammaModel::quantile_normal(double p) const {
    // Helper function - standard normal quantile
    if (p < 0.5) {
        double q = std::sqrt(-2.0 * std::log(p));
        return -(2.515517 + 0.802853 * q + 0.010328 * q * q) /
               (1.0 + 1.432788 * q + 0.189269 * q * q + 0.001308 * q * q * q);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return (2.515517 + 0.802853 * q + 0.010328 * q * q) /
               (1.0 + 1.432788 * q + 0.189269 * q * q + 0.001308 * q * q * q);
    }
}