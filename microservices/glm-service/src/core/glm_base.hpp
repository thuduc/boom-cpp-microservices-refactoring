#ifndef GLM_BASE_HPP
#define GLM_BASE_HPP

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

struct FitResult {
    Eigen::VectorXd coefficients;
    double intercept;
    int iterations;
    bool converged;
    double log_likelihood;
    double aic;
    double bic;
    Eigen::VectorXd standard_errors;
    Eigen::VectorXd p_values;
    std::vector<std::pair<double, double>> confidence_intervals;
};

class GLMBase {
public:
    virtual ~GLMBase() = default;
    
    // Fitting
    virtual FitResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
    
    // Prediction
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;
    virtual Eigen::VectorXd predictLinear(const Eigen::MatrixXd& X) const;
    virtual Eigen::VectorXd predictProbability(const Eigen::MatrixXd& X) const { 
        return predict(X); 
    }
    
    // Model configuration
    void setFitIntercept(bool fit) { fit_intercept_ = fit; }
    void setStandardize(bool standardize) { standardize_ = standardize; }
    void setRegularization(double lambda, const std::string& type);
    void setWeights(const Eigen::VectorXd& weights) { weights_ = weights; }
    
    // Parameter management
    void setCoefficients(const Eigen::VectorXd& coef, double intercept);
    Eigen::VectorXd getCoefficients() const { return coefficients_; }
    double getIntercept() const { return intercept_; }
    
    // Prediction intervals
    virtual json predictionIntervals(const Eigen::MatrixXd& X, double alpha = 0.05) const;
    
protected:
    // Model parameters
    Eigen::VectorXd coefficients_;
    double intercept_ = 0.0;
    
    // Configuration
    bool fit_intercept_ = true;
    bool standardize_ = false;
    double regularization_lambda_ = 0.0;
    std::string regularization_type_ = "l2";
    Eigen::VectorXd weights_;
    
    // Standardization parameters
    Eigen::VectorXd mean_;
    Eigen::VectorXd std_;
    
    // Helper methods
    Eigen::MatrixXd addIntercept(const Eigen::MatrixXd& X) const;
    void standardizeData(Eigen::MatrixXd& X, Eigen::VectorXd& y);
    void unstandardizeCoefficients();
    
    // Information criteria
    double computeAIC(double log_likelihood, int n_params) const;
    double computeBIC(double log_likelihood, int n_params, int n_obs) const;
    
    // Link functions - to be overridden by specific models
    virtual double linkFunction(double linear_pred) const = 0;
    virtual double linkDerivative(double linear_pred) const = 0;
    virtual double inverseLinkFunction(double eta) const = 0;
    
    // Log likelihood computation
    virtual double computeLogLikelihood(const Eigen::MatrixXd& X, 
                                       const Eigen::VectorXd& y) const = 0;
};

#endif // GLM_BASE_HPP