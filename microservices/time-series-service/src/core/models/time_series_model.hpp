#ifndef TIME_SERIES_MODEL_HPP
#define TIME_SERIES_MODEL_HPP

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

struct FitResult {
    json parameters;
    double log_likelihood;
    double aic;
    double bic;
    bool converged;
    int iterations;
    std::vector<double> residuals;
    std::vector<std::vector<double>> state_estimates;
};

struct ForecastResult {
    std::vector<double> point_forecast;
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::vector<double> forecast_errors;
};

class TimeSeriesModel {
public:
    virtual ~TimeSeriesModel() = default;
    
    // Fitting
    virtual FitResult fit(const std::vector<double>& data, 
                         const std::vector<double>& timestamps = {}) = 0;
    
    // Forecasting
    virtual ForecastResult forecast(int horizon,
                                   const std::vector<double>& historical_data = {},
                                   double confidence = 0.95) = 0;
    
    // Parameter management
    virtual void setParameters(const json& params) = 0;
    virtual json getParameters() const = 0;
    
    // State space representation
    virtual Eigen::MatrixXd getTransitionMatrix() const = 0;
    virtual Eigen::MatrixXd getObservationMatrix() const = 0;
    virtual Eigen::MatrixXd getStateCovariance() const = 0;
    virtual Eigen::MatrixXd getObservationCovariance() const = 0;
    
protected:
    // Helper methods
    double computeAIC(double log_likelihood, int n_params) const {
        return -2.0 * log_likelihood + 2.0 * n_params;
    }
    
    double computeBIC(double log_likelihood, int n_params, int n_obs) const {
        return -2.0 * log_likelihood + n_params * std::log(n_obs);
    }
};

#endif // TIME_SERIES_MODEL_HPP