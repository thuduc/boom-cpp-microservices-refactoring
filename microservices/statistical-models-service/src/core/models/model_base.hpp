#ifndef MODEL_BASE_HPP
#define MODEL_BASE_HPP

#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct FitResult {
    json parameters;
    double log_likelihood;
    bool converged;
    int iterations;
    std::vector<double> standard_errors;
    std::vector<std::pair<double, double>> confidence_intervals;
};

class StatisticalModel {
public:
    virtual ~StatisticalModel() = default;
    
    // Fitting methods
    virtual FitResult fitMLE(const std::vector<double>& data) = 0;
    virtual FitResult fitBayesian(const std::vector<double>& data, const json& prior_params) = 0;
    
    // Parameter management
    virtual void setParameters(const json& params) = 0;
    virtual json getParameters() const = 0;
    virtual int getNumParameters() const = 0;
    
    // Distribution properties
    virtual double mean() const = 0;
    virtual double variance() const = 0;
    virtual double quantile(double p) const = 0;
    
    // Probability functions
    virtual double pdf(double x) const = 0;
    virtual double cdf(double x) const = 0;
    virtual double logLikelihood(const std::vector<double>& data) const = 0;
    
    // Simulation
    virtual std::vector<double> simulate(int n_samples) const = 0;
    
protected:
    // Helper for numerical optimization
    static double negativeLLogLikelihood(const std::vector<double>& params, 
                                       const std::vector<double>& data,
                                       StatisticalModel* model);
};

#endif // MODEL_BASE_HPP