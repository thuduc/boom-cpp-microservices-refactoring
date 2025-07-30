#ifndef GAUSSIAN_MODEL_HPP
#define GAUSSIAN_MODEL_HPP

#include "model_base.hpp"
#include <random>

class GaussianModel : public StatisticalModel {
public:
    GaussianModel();
    
    // Fitting methods
    FitResult fitMLE(const std::vector<double>& data) override;
    FitResult fitBayesian(const std::vector<double>& data, const json& prior_params) override;
    
    // Parameter management
    void setParameters(const json& params) override;
    json getParameters() const override;
    int getNumParameters() const override { return 2; }
    
    // Distribution properties
    double mean() const override { return mu_; }
    double variance() const override { return sigma_ * sigma_; }
    double quantile(double p) const override;
    
    // Probability functions
    double pdf(double x) const override;
    double cdf(double x) const override;
    double logLikelihood(const std::vector<double>& data) const override;
    
    // Simulation
    std::vector<double> simulate(int n_samples) const override;
    
private:
    double mu_;     // mean
    double sigma_;  // standard deviation
    
    mutable std::mt19937 rng_;
};

#endif // GAUSSIAN_MODEL_HPP