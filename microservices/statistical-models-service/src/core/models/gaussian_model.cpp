#include "gaussian_model.hpp"
#include "../estimation/mle_estimator.hpp"
#include "../estimation/bayesian_estimator.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

GaussianModel::GaussianModel() : mu_(0.0), sigma_(1.0), rng_(std::random_device{}()) {}

FitResult GaussianModel::fitMLE(const std::vector<double>& data) {
    FitResult result;
    
    if (data.empty()) {
        result.converged = false;
        return result;
    }
    
    // MLE for Gaussian: sample mean and standard deviation
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    mu_ = sum / data.size();
    
    double sum_sq_diff = 0.0;
    for (double x : data) {
        sum_sq_diff += (x - mu_) * (x - mu_);
    }
    sigma_ = std::sqrt(sum_sq_diff / data.size());
    
    // Compute log likelihood
    result.log_likelihood = logLikelihood(data);
    
    // Store parameters
    result.parameters = {
        {"mean", mu_},
        {"std_dev", sigma_}
    };
    
    // Compute standard errors (asymptotic)
    double n = data.size();
    result.standard_errors = {
        sigma_ / std::sqrt(n),  // SE of mean
        sigma_ / std::sqrt(2 * n)  // SE of std dev
    };
    
    // 95% confidence intervals
    double z = 1.96;  // 95% CI
    result.confidence_intervals = {
        {mu_ - z * result.standard_errors[0], mu_ + z * result.standard_errors[0]},
        {sigma_ - z * result.standard_errors[1], sigma_ + z * result.standard_errors[1]}
    };
    
    result.converged = true;
    result.iterations = 1;  // Closed form solution
    
    return result;
}

FitResult GaussianModel::fitBayesian(const std::vector<double>& data, const json& prior_params) {
    FitResult result;
    
    // Normal-Inverse-Gamma conjugate prior
    double prior_mu = prior_params.value("mean", 0.0);
    double prior_kappa = prior_params.value("kappa", 1.0);
    double prior_alpha = prior_params.value("alpha", 1.0);
    double prior_beta = prior_params.value("beta", 1.0);
    
    int n = data.size();
    double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    
    double sum_sq = 0.0;
    for (double x : data) {
        sum_sq += x * x;
    }
    
    // Posterior parameters
    double post_kappa = prior_kappa + n;
    double post_mu = (prior_kappa * prior_mu + n * sample_mean) / post_kappa;
    double post_alpha = prior_alpha + n / 2.0;
    double post_beta = prior_beta + 0.5 * sum_sq + 
                       0.5 * prior_kappa * n * (sample_mean - prior_mu) * (sample_mean - prior_mu) / post_kappa;
    
    // Posterior mean estimates
    mu_ = post_mu;
    sigma_ = std::sqrt(post_beta / (post_alpha - 1));  // Mode of inverse gamma
    
    result.parameters = {
        {"mean", mu_},
        {"std_dev", sigma_},
        {"posterior", {
            {"mu", post_mu},
            {"kappa", post_kappa},
            {"alpha", post_alpha},
            {"beta", post_beta}
        }}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;  // Conjugate prior has closed form
    
    return result;
}

void GaussianModel::setParameters(const json& params) {
    mu_ = params.value("mean", 0.0);
    sigma_ = params.value("std_dev", 1.0);
}

json GaussianModel::getParameters() const {
    return {
        {"mean", mu_},
        {"std_dev", sigma_}
    };
}

double GaussianModel::quantile(double p) const {
    // Using inverse error function approximation
    if (p <= 0) return -std::numeric_limits<double>::infinity();
    if (p >= 1) return std::numeric_limits<double>::infinity();
    
    // Approximate inverse normal CDF
    double z;
    if (p < 0.5) {
        double q = std::sqrt(-2.0 * std::log(p));
        z = -(2.515517 + 0.802853 * q + 0.010328 * q * q) /
            (1.0 + 1.432788 * q + 0.189269 * q * q + 0.001308 * q * q * q);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        z = (2.515517 + 0.802853 * q + 0.010328 * q * q) /
            (1.0 + 1.432788 * q + 0.189269 * q * q + 0.001308 * q * q * q);
    }
    
    return mu_ + sigma_ * z;
}

double GaussianModel::pdf(double x) const {
    double z = (x - mu_) / sigma_;
    return (1.0 / (sigma_ * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
}

double GaussianModel::cdf(double x) const {
    double z = (x - mu_) / (sigma_ * std::sqrt(2.0));
    return 0.5 * (1.0 + std::erf(z));
}

double GaussianModel::logLikelihood(const std::vector<double>& data) const {
    double log_lik = 0.0;
    double log_norm = -0.5 * std::log(2.0 * M_PI) - std::log(sigma_);
    
    for (double x : data) {
        double z = (x - mu_) / sigma_;
        log_lik += log_norm - 0.5 * z * z;
    }
    
    return log_lik;
}

std::vector<double> GaussianModel::simulate(int n_samples) const {
    std::normal_distribution<> dist(mu_, sigma_);
    std::vector<double> samples(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        samples[i] = dist(rng_);
    }
    
    return samples;
}