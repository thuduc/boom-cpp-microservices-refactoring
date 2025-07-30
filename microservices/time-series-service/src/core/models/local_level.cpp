#include "local_level.hpp"
#include "../filters/kalman_filter.hpp"
#include <numeric>
#include <cmath>

LocalLevelModel::LocalLevelModel() 
    : level_(0.0), sigma_obs_(1.0), sigma_level_(0.1) {}

void LocalLevelModel::estimateParametersEM(const std::vector<double>& data) {
    // EM algorithm for parameter estimation
    const int max_iter = 50;
    const double tol = 1e-6;
    
    // Initialize with simple estimates
    level_ = data[0];
    double var = 0.0;
    for (size_t i = 1; i < data.size(); ++i) {
        var += (data[i] - data[i-1]) * (data[i] - data[i-1]);
    }
    var /= (data.size() - 1);
    sigma_obs_ = std::sqrt(var * 0.7);
    sigma_level_ = std::sqrt(var * 0.3);
    
    KalmanFilter kf;
    double prev_log_lik = -std::numeric_limits<double>::infinity();
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // E-step: Run Kalman filter and smoother
        auto filtered = kf.filter(this, data);
        auto smoothed = kf.smooth(this, data);
        
        // M-step: Update parameters
        double sum_v_obs = 0.0;
        double sum_v_level = 0.0;
        size_t n = data.size();
        
        for (size_t t = 0; t < n; ++t) {
            double obs_error = data[t] - smoothed.smoothed_states[t][0];
            sum_v_obs += obs_error * obs_error + smoothed.smoothed_variances[t](0, 0);
            
            if (t > 0) {
                double level_error = smoothed.smoothed_states[t][0] - smoothed.smoothed_states[t-1][0];
                sum_v_level += level_error * level_error;
            }
        }
        
        sigma_obs_ = std::sqrt(sum_v_obs / n);
        sigma_level_ = std::sqrt(sum_v_level / (n - 1));
        
        // Check convergence
        double log_lik = computeLogLikelihood(data);
        if (std::abs(log_lik - prev_log_lik) < tol) {
            break;
        }
        prev_log_lik = log_lik;
    }
}

double LocalLevelModel::computeLogLikelihood(const std::vector<double>& data) const {
    KalmanFilter kf;
    auto result = kf.filter(this, data);
    
    double log_lik = 0.0;
    for (size_t t = 0; t < data.size(); ++t) {
        double innovation = result.innovations[t];
        double innovation_var = result.filtered_variances[t](0, 0) + sigma_obs_ * sigma_obs_;
        
        log_lik -= 0.5 * (std::log(2 * M_PI * innovation_var) + 
                         innovation * innovation / innovation_var);
    }
    
    return log_lik;
}

FitResult LocalLevelModel::fit(const std::vector<double>& data, 
                              const std::vector<double>& timestamps) {
    FitResult result;
    
    // Estimate parameters using EM algorithm
    estimateParametersEM(data);
    
    // Run final Kalman filter to get states
    KalmanFilter kf;
    auto smoothed = kf.smooth(this, data);
    
    // Extract level estimates
    result.state_estimates.resize(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        result.state_estimates[t] = {smoothed.smoothed_states[t][0]};
    }
    
    // Compute residuals
    result.residuals.resize(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        result.residuals[t] = data[t] - result.state_estimates[t][0];
    }
    
    // Final level estimate
    level_ = result.state_estimates.back()[0];
    
    // Compute fit statistics
    result.log_likelihood = computeLogLikelihood(data);
    int n_params = 2;  // sigma_obs, sigma_level
    result.aic = computeAIC(result.log_likelihood, n_params);
    result.bic = computeBIC(result.log_likelihood, n_params, data.size());
    result.converged = true;
    result.iterations = 1;  // Simplified
    
    result.parameters = {
        {"level", level_},
        {"sigma_obs", sigma_obs_},
        {"sigma_level", sigma_level_}
    };
    
    return result;
}

ForecastResult LocalLevelModel::forecast(int horizon,
                                        const std::vector<double>& historical_data,
                                        double confidence) {
    ForecastResult result;
    result.point_forecast.resize(horizon);
    result.lower_bound.resize(horizon);
    result.upper_bound.resize(horizon);
    result.forecast_errors.resize(horizon);
    
    // Initialize level from historical data if provided
    double current_level = level_;
    if (!historical_data.empty()) {
        // Re-estimate using historical data
        FitResult fit_result = fit(historical_data);
        current_level = fit_result.parameters["level"];
    }
    
    // Z-score for confidence interval
    double z_score = 1.96;  // 95% confidence
    if (confidence == 0.99) z_score = 2.576;
    else if (confidence == 0.90) z_score = 1.645;
    
    // Generate forecasts
    for (int h = 0; h < horizon; ++h) {
        // Point forecast (constant level)
        result.point_forecast[h] = current_level;
        
        // Forecast variance increases with horizon
        double forecast_var = sigma_obs_ * sigma_obs_ + 
                             (h + 1) * sigma_level_ * sigma_level_;
        result.forecast_errors[h] = std::sqrt(forecast_var);
        
        // Prediction intervals
        result.lower_bound[h] = current_level - z_score * result.forecast_errors[h];
        result.upper_bound[h] = current_level + z_score * result.forecast_errors[h];
    }
    
    return result;
}

void LocalLevelModel::setParameters(const json& params) {
    if (params.contains("level")) level_ = params["level"];
    if (params.contains("sigma_obs")) sigma_obs_ = params["sigma_obs"];
    if (params.contains("sigma_level")) sigma_level_ = params["sigma_level"];
}

json LocalLevelModel::getParameters() const {
    return {
        {"level", level_},
        {"sigma_obs", sigma_obs_},
        {"sigma_level", sigma_level_}
    };
}

Eigen::MatrixXd LocalLevelModel::getTransitionMatrix() const {
    Eigen::MatrixXd T(1, 1);
    T(0, 0) = 1.0;  // Level follows random walk
    return T;
}

Eigen::MatrixXd LocalLevelModel::getObservationMatrix() const {
    Eigen::MatrixXd Z(1, 1);
    Z(0, 0) = 1.0;
    return Z;
}

Eigen::MatrixXd LocalLevelModel::getStateCovariance() const {
    Eigen::MatrixXd Q(1, 1);
    Q(0, 0) = sigma_level_ * sigma_level_;
    return Q;
}

Eigen::MatrixXd LocalLevelModel::getObservationCovariance() const {
    Eigen::MatrixXd H(1, 1);
    H(0, 0) = sigma_obs_ * sigma_obs_;
    return H;
}