#ifndef MIXTURE_BASE_HPP
#define MIXTURE_BASE_HPP

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

struct FitResult {
    int n_components;
    double log_likelihood;
    double bic;
    double aic;
    bool converged;
    int iterations;
    std::vector<double> weights;
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::MatrixXd> covariances;
};

struct SampleResult {
    Eigen::MatrixXd samples;
    std::vector<int> component_labels;
};

class MixtureModel {
public:
    virtual ~MixtureModel() = default;
    
    // Fitting methods
    virtual FitResult fitEM(const Eigen::MatrixXd& data, int max_iter = 100) = 0;
    virtual FitResult fitVariational(const Eigen::MatrixXd& data, int max_iter = 100) = 0;
    
    // Prediction
    virtual std::vector<int> predict(const Eigen::MatrixXd& data) = 0;
    virtual Eigen::MatrixXd predictProba(const Eigen::MatrixXd& data) = 0;
    virtual std::vector<double> computeDensity(const Eigen::MatrixXd& data) = 0;
    
    // Sampling
    virtual SampleResult sample(int n_samples, bool return_components = false) = 0;
    
    // Parameter management
    virtual void setParameters(const json& params) = 0;
    virtual json getParameters() const = 0;
    
protected:
    // Helper methods
    double computeBIC(double log_likelihood, int n_params, int n_samples) const {
        return -2.0 * log_likelihood + n_params * std::log(n_samples);
    }
    
    double computeAIC(double log_likelihood, int n_params) const {
        return -2.0 * log_likelihood + 2.0 * n_params;
    }
};

#endif // MIXTURE_BASE_HPP