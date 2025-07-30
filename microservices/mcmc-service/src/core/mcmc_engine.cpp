#include "mcmc_engine.hpp"

std::mt19937& MCMCEngine::getRNG() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

std::vector<double> MCMCEngine::normalProposal(
    const std::vector<double>& current,
    double scale) {
    
    std::normal_distribution<> normal(0.0, scale);
    std::vector<double> proposed = current;
    
    for (size_t i = 0; i < current.size(); ++i) {
        proposed[i] += normal(getRNG());
    }
    
    return proposed;
}

std::vector<double> MCMCEngine::uniformProposal(
    const std::vector<double>& current,
    double scale) {
    
    std::uniform_real_distribution<> uniform(-scale, scale);
    std::vector<double> proposed = current;
    
    for (size_t i = 0; i < current.size(); ++i) {
        proposed[i] += uniform(getRNG());
    }
    
    return proposed;
}