#include "multinomial_model.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

MultinomialModel::MultinomialModel() : trials_(1), rng_(std::random_device{}()) {
    probabilities_ = {0.5, 0.5};  // Default to fair coin
}

double MultinomialModel::mean() const {
    // Return mean of first category
    if (probabilities_.empty()) return 0.0;
    return trials_ * probabilities_[0];
}

double MultinomialModel::variance() const {
    // Return variance of first category
    if (probabilities_.empty()) return 0.0;
    return trials_ * probabilities_[0] * (1 - probabilities_[0]);
}

FitResult MultinomialModel::fitMLE(const std::vector<double>& data) {
    FitResult result;
    
    if (data.empty()) {
        result.converged = false;
        return result;
    }
    
    // Count occurrences of each category
    // Assuming data contains category indices (0, 1, 2, ...)
    int max_category = 0;
    for (double x : data) {
        max_category = std::max(max_category, static_cast<int>(x));
    }
    
    std::vector<int> counts(max_category + 1, 0);
    for (double x : data) {
        int category = static_cast<int>(x);
        if (category >= 0 && category < counts.size()) {
            counts[category]++;
        }
    }
    
    // MLE: relative frequencies
    probabilities_.resize(counts.size());
    int total = data.size();
    for (size_t i = 0; i < counts.size(); ++i) {
        probabilities_[i] = static_cast<double>(counts[i]) / total;
    }
    
    // Store as JSON array
    result.parameters = {
        {"probabilities", probabilities_}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;
    
    // Standard errors (multinomial)
    result.standard_errors.resize(probabilities_.size());
    for (size_t i = 0; i < probabilities_.size(); ++i) {
        result.standard_errors[i] = std::sqrt(probabilities_[i] * (1 - probabilities_[i]) / total);
    }
    
    return result;
}

FitResult MultinomialModel::fitBayesian(const std::vector<double>& data, const json& prior_params) {
    FitResult result;
    
    // Dirichlet conjugate prior
    std::vector<double> prior_alpha = prior_params.value("alpha", 
        std::vector<double>(probabilities_.size(), 1.0));
    
    // Count data
    std::vector<int> counts(prior_alpha.size(), 0);
    for (double x : data) {
        int category = static_cast<int>(x);
        if (category >= 0 && category < counts.size()) {
            counts[category]++;
        }
    }
    
    // Posterior parameters
    std::vector<double> post_alpha(prior_alpha.size());
    double sum_post_alpha = 0.0;
    for (size_t i = 0; i < prior_alpha.size(); ++i) {
        post_alpha[i] = prior_alpha[i] + counts[i];
        sum_post_alpha += post_alpha[i];
    }
    
    // Posterior mean
    probabilities_.resize(post_alpha.size());
    for (size_t i = 0; i < post_alpha.size(); ++i) {
        probabilities_[i] = post_alpha[i] / sum_post_alpha;
    }
    
    result.parameters = {
        {"probabilities", probabilities_},
        {"posterior_alpha", post_alpha}
    };
    
    result.log_likelihood = logLikelihood(data);
    result.converged = true;
    result.iterations = 1;
    
    return result;
}

void MultinomialModel::setParameters(const json& params) {
    if (params.contains("probabilities")) {
        probabilities_ = params["probabilities"].get<std::vector<double>>();
        
        // Normalize to ensure sum = 1
        double sum = std::accumulate(probabilities_.begin(), probabilities_.end(), 0.0);
        if (sum > 0) {
            for (double& p : probabilities_) {
                p /= sum;
            }
        }
    }
    
    if (params.contains("trials")) {
        trials_ = params["trials"];
    }
}

json MultinomialModel::getParameters() const {
    return {
        {"probabilities", probabilities_},
        {"trials", trials_}
    };
}

double MultinomialModel::quantile(double p) const {
    // For categorical data, return category index
    double cumsum = 0.0;
    for (size_t i = 0; i < probabilities_.size(); ++i) {
        cumsum += probabilities_[i];
        if (cumsum >= p) {
            return static_cast<double>(i);
        }
    }
    return static_cast<double>(probabilities_.size() - 1);
}

double MultinomialModel::pdf(double x) const {
    int category = static_cast<int>(x);
    if (category < 0 || category >= probabilities_.size()) return 0.0;
    
    // For single observation
    return probabilities_[category];
}

double MultinomialModel::cdf(double x) const {
    int max_category = static_cast<int>(std::floor(x));
    if (max_category < 0) return 0.0;
    
    double cumsum = 0.0;
    for (int i = 0; i <= max_category && i < probabilities_.size(); ++i) {
        cumsum += probabilities_[i];
    }
    
    return cumsum;
}

double MultinomialModel::logLikelihood(const std::vector<double>& data) const {
    double log_lik = 0.0;
    
    for (double x : data) {
        int category = static_cast<int>(x);
        if (category < 0 || category >= probabilities_.size()) {
            return -std::numeric_limits<double>::infinity();
        }
        
        log_lik += std::log(probabilities_[category]);
    }
    
    return log_lik;
}

std::vector<double> MultinomialModel::simulate(int n_samples) const {
    std::discrete_distribution<> dist(probabilities_.begin(), probabilities_.end());
    std::vector<double> samples(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        samples[i] = static_cast<double>(dist(rng_));
    }
    
    return samples;
}

std::vector<int> MultinomialModel::simulateMultinomial(int n_samples) const {
    // Simulate multinomial counts for given number of trials
    std::discrete_distribution<> dist(probabilities_.begin(), probabilities_.end());
    std::vector<int> counts(probabilities_.size(), 0);
    
    for (int i = 0; i < trials_; ++i) {
        counts[dist(rng_)]++;
    }
    
    return counts;
}