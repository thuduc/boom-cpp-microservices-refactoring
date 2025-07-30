#ifndef BETA_MODEL_HPP
#define BETA_MODEL_HPP

#include "model_base.hpp"
#include <random>

class BetaModel : public StatisticalModel {
public:
    BetaModel();
    
    // Fitting methods
    FitResult fitMLE(const std::vector<double>& data) override;
    FitResult fitBayesian(const std::vector<double>& data, const json& prior_params) override;
    
    // Parameter management
    void setParameters(const json& params) override;
    json getParameters() const override;
    int getNumParameters() const override { return 2; }
    
    // Distribution properties
    double mean() const override { return alpha_ / (alpha_ + beta_); }
    double variance() const override;
    double quantile(double p) const override;
    
    // Probability functions
    double pdf(double x) const override;
    double cdf(double x) const override;
    double logLikelihood(const std::vector<double>& data) const override;
    
    // Simulation
    std::vector<double> simulate(int n_samples) const override;
    
private:
    double alpha_;
    double beta_;
    
    mutable std::mt19937 rng_;
    
    static double trigamma(double x);
};

#endif // BETA_MODEL_HPP