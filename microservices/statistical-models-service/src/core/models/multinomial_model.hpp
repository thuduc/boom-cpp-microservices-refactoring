#ifndef MULTINOMIAL_MODEL_HPP
#define MULTINOMIAL_MODEL_HPP

#include "model_base.hpp"
#include <random>

class MultinomialModel : public StatisticalModel {
public:
    MultinomialModel();
    
    // Fitting methods
    FitResult fitMLE(const std::vector<double>& data) override;
    FitResult fitBayesian(const std::vector<double>& data, const json& prior_params) override;
    
    // Parameter management
    void setParameters(const json& params) override;
    json getParameters() const override;
    int getNumParameters() const override { return probabilities_.size(); }
    
    // Distribution properties (returns first category mean as example)
    double mean() const override;
    double variance() const override;
    double quantile(double p) const override;
    
    // Probability functions
    double pdf(double x) const override;
    double cdf(double x) const override;
    double logLikelihood(const std::vector<double>& data) const override;
    
    // Simulation
    std::vector<double> simulate(int n_samples) const override;
    
    // Multinomial-specific methods
    void setTrials(int n) { trials_ = n; }
    std::vector<int> simulateMultinomial(int n_samples) const;
    
private:
    std::vector<double> probabilities_;
    int trials_;
    
    mutable std::mt19937 rng_;
};

#endif // MULTINOMIAL_MODEL_HPP