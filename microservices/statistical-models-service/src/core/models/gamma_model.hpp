#ifndef GAMMA_MODEL_HPP
#define GAMMA_MODEL_HPP

#include "model_base.hpp"
#include <random>

class GammaModel : public StatisticalModel {
public:
    GammaModel();
    
    // Fitting methods
    FitResult fitMLE(const std::vector<double>& data) override;
    FitResult fitBayesian(const std::vector<double>& data, const json& prior_params) override;
    
    // Parameter management
    void setParameters(const json& params) override;
    json getParameters() const override;
    int getNumParameters() const override { return 2; }
    
    // Distribution properties
    double mean() const override { return shape_ * scale_; }
    double variance() const override { return shape_ * scale_ * scale_; }
    double quantile(double p) const override;
    
    // Probability functions
    double pdf(double x) const override;
    double cdf(double x) const override;
    double logLikelihood(const std::vector<double>& data) const override;
    
    // Simulation
    std::vector<double> simulate(int n_samples) const override;
    
private:
    double shape_;  // alpha
    double scale_;  // beta (scale parameter)
    
    mutable std::mt19937 rng_;
    
    // Helper for MLE
    static double digamma(double x);
    static double trigamma(double x);
    double quantile_normal(double p) const;
};

#endif // GAMMA_MODEL_HPP