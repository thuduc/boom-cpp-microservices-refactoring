#include "poisson_model.hpp"
#include <cmath>
#include <numeric>

PoissonModel::PoissonModel() : lambda_(1.0), rng_(std::random_device{}()) {}

FitResult PoissonModel::fitMLE(const std::vector<double>& data) {
    FitResult result;
    
    if (data.empty()) {
        result.converged = false;
        return result;
    }
    
    // MLE for Poisson is the sample mean
    lambda_ = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    
    result.parameters = {
        {"lambda", lambda_}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;  // Closed form solution
    
    // Standard error
    double n = data.size();
    double se = std::sqrt(lambda_ / n);
    result.standard_errors = {se};
    
    // 95% confidence interval
    double z = 1.96;
    result.confidence_intervals = {
        {lambda_ - z * se, lambda_ + z * se}
    };
    
    return result;
}

FitResult PoissonModel::fitBayesian(const std::vector<double>& data, const json& prior_params) {
    FitResult result;
    
    // Gamma conjugate prior
    double prior_alpha = prior_params.value("alpha", 1.0);
    double prior_beta = prior_params.value("beta", 1.0);
    
    // Posterior parameters
    double sum_data = std::accumulate(data.begin(), data.end(), 0.0);
    double n = data.size();
    
    double post_alpha = prior_alpha + sum_data;
    double post_beta = prior_beta + n;
    
    // Posterior mean
    lambda_ = post_alpha / post_beta;
    
    result.parameters = {
        {"lambda", lambda_},
        {"posterior", {
            {"alpha", post_alpha},
            {"beta", post_beta}
        }}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;  // Conjugate prior has closed form
    
    return result;
}

void PoissonModel::setParameters(const json& params) {
    lambda_ = params.value("lambda", 1.0);
}

json PoissonModel::getParameters() const {
    return {
        {"lambda", lambda_}
    };
}

double PoissonModel::quantile(double p) const {
    if (p <= 0) return 0.0;
    if (p >= 1) return std::numeric_limits<double>::infinity();
    
    // Find quantile by searching
    double cumulative = 0.0;
    int k = 0;
    
    while (cumulative < p) {
        cumulative += pdf(k);
        if (cumulative >= p) break;
        k++;
        
        // Safety check
        if (k > 1000 * lambda_) break;
    }
    
    return static_cast<double>(k);
}

double PoissonModel::pdf(double x) const {
    int k = static_cast<int>(std::round(x));
    if (k < 0) return 0.0;
    
    // Use log for numerical stability
    double log_pmf = k * std::log(lambda_) - lambda_ - std::lgamma(k + 1);
    return std::exp(log_pmf);
}

double PoissonModel::cdf(double x) const {
    if (x < 0) return 0.0;
    
    int k_max = static_cast<int>(std::floor(x));
    double cumulative = 0.0;
    
    for (int k = 0; k <= k_max; ++k) {
        cumulative += pdf(k);
    }
    
    return cumulative;
}

double PoissonModel::logLikelihood(const std::vector<double>& data) const {
    double log_lik = 0.0;
    
    for (double x : data) {
        int k = static_cast<int>(std::round(x));
        if (k < 0) return -std::numeric_limits<double>::infinity();
        
        log_lik += k * std::log(lambda_) - lambda_ - std::lgamma(k + 1);
    }
    
    return log_lik;
}

std::vector<double> PoissonModel::simulate(int n_samples) const {
    std::poisson_distribution<> dist(lambda_);
    std::vector<double> samples(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        samples[i] = static_cast<double>(dist(rng_));
    }
    
    return samples;
}