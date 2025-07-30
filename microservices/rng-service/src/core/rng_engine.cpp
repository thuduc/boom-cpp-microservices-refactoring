#include "core/rng_engine.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

RNGEngine& RNGEngine::getInstance() {
    static RNGEngine instance;
    return instance;
}

RNGEngine::RNGEngine() : generator_(std::random_device{}()) {}

void RNGEngine::seed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    generator_.seed(seed);
}

std::vector<double> RNGEngine::uniform(int n, double min, double max) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uniform_real_distribution<double> dist(min, max);
    
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = dist(generator_);
    }
    return result;
}

std::vector<double> RNGEngine::normal(int n, double mean, double sd) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::normal_distribution<double> dist(mean, sd);
    
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = dist(generator_);
    }
    return result;
}

std::vector<double> RNGEngine::gamma(int n, double shape, double scale) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::gamma_distribution<double> dist(shape, scale);
    
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = dist(generator_);
    }
    return result;
}

std::vector<double> RNGEngine::beta(int n, double a, double b) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Beta distribution using gamma variates
    std::gamma_distribution<double> gamma_a(a, 1.0);
    std::gamma_distribution<double> gamma_b(b, 1.0);
    
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        double x = gamma_a(generator_);
        double y = gamma_b(generator_);
        result[i] = x / (x + y);
    }
    return result;
}

std::vector<std::vector<int>> RNGEngine::multinomial(int n, int trials, 
                                                      const std::vector<double>& probs) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::vector<int>> result(n);
    
    for (int i = 0; i < n; ++i) {
        std::vector<int> counts(probs.size(), 0);
        int remaining = trials;
        
        for (size_t j = 0; j < probs.size() - 1 && remaining > 0; ++j) {
            // Compute conditional probability
            double sum_remaining = std::accumulate(probs.begin() + j, probs.end(), 0.0);
            double p = probs[j] / sum_remaining;
            
            std::binomial_distribution<int> binom(remaining, p);
            counts[j] = binom(generator_);
            remaining -= counts[j];
        }
        
        // Last category gets remaining trials
        if (probs.size() > 0) {
            counts[probs.size() - 1] = remaining;
        }
        
        result[i] = counts;
    }
    
    return result;
}