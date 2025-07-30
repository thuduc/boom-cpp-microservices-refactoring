#ifndef POISSON_MODEL_HPP
#define POISSON_MODEL_HPP

#include "model_base.hpp"
#include <random>

class PoissonModel : public StatisticalModel {
public:
    PoissonModel();
    
    // Fitting methods
    FitResult fitMLE(const std::vector<double>& data) override;
    FitResult fitBayesian(const std::vector<double>& data, const json& prior_params) override;
    
    // Parameter management
    void setParameters(const json& params) override;
    json getParameters() const override;
    int getNumParameters() const override { return 1; }
    
    // Distribution properties
    double mean() const override { return lambda_; }
    double variance() const override { return lambda_; }
    double quantile(double p) const override;
    
    // Probability functions
    double pdf(double x) const override;
    double cdf(double x) const override;
    double logLikelihood(const std::vector<double>& data) const override;
    
    // Simulation
    std::vector<double> simulate(int n_samples) const override;
    
private:
    double lambda_;  // rate parameter
    
    mutable std::mt19937 rng_;
};

#endif // POISSON_MODEL_HPP