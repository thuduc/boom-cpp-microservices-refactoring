#include "local_linear_trend.hpp"
#include "../filters/kalman_filter.hpp"
#include <numeric>
#include <cmath>

LocalLinearTrendModel::LocalLinearTrendModel() 
    : level_(0.0), trend_(0.0), sigma_obs_(1.0), sigma_level_(0.1), sigma_trend_(0.01) {}

void LocalLinearTrendModel::estimateParametersEM(const std::vector<double>& data) {
    // EM algorithm for parameter estimation
    const int max_iter = 50;
    const double tol = 1e-6;
    
    // Initialize with simple estimates
    level_ = data[0];
    trend_ = (data[data.size()-1] - data[0]) / (data.size() - 1);
    
    double var = 0.0;
    for (size_t i = 1; i < data.size(); ++i) {
        var += (data[i] - data[i-1]) * (data[i] - data[i-1]);
    }
    var /= (data.size() - 1);
    
    sigma_obs_ = std::sqrt(var * 0.6);
    sigma_level_ = std::sqrt(var * 0.3);
    sigma_trend_ = std::sqrt(var * 0.1);
    
    KalmanFilter kf;
    double prev_log_lik = -std::numeric_limits<double>::infinity();
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // E-step: Run Kalman filter and smoother
        auto smoothed = kf.smooth(this, data);
        
        // M-step: Update parameters
        double sum_v_obs = 0.0;
        double sum_v_level = 0.0;
        double sum_v_trend = 0.0;
        size_t n = data.size();
        
        for (size_t t = 0; t < n; ++t) {
            // Observation error
            double obs_error = data[t] - smoothed.smoothed_states[t][0];
            sum_v_obs += obs_error * obs_error + smoothed.smoothed_variances[t](0, 0);
            
            if (t > 0) {
                // Level innovation
                double expected_level = smoothed.smoothed_states[t-1][0] + smoothed.smoothed_states[t-1][1];
                double level_error = smoothed.smoothed_states[t][0] - expected_level;
                sum_v_level += level_error * level_error;
                
                // Trend innovation
                double trend_error = smoothed.smoothed_states[t][1] - smoothed.smoothed_states[t-1][1];
                sum_v_trend += trend_error * trend_error;
            }
        }
        
        sigma_obs_ = std::sqrt(sum_v_obs / n);
        sigma_level_ = std::sqrt(sum_v_level / (n - 1));
        sigma_trend_ = std::sqrt(sum_v_trend / (n - 1));
        
        // Check convergence
        double log_lik = computeLogLikelihood(data);
        if (std::abs(log_lik - prev_log_lik) < tol) {
            break;
        }
        prev_log_lik = log_lik;
    }
}

double LocalLinearTrendModel::computeLogLikelihood(const std::vector<double>& data) const {
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

FitResult LocalLinearTrendModel::fit(const std::vector<double>& data, 
                                    const std::vector<double>& timestamps) {
    FitResult result;
    
    // Estimate parameters
    estimateParametersEM(data);
    
    // Run final Kalman smoother
    KalmanFilter kf;
    auto smoothed = kf.smooth(this, data);
    
    // Extract state estimates
    result.state_estimates.resize(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        result.state_estimates[t] = {
            smoothed.smoothed_states[t][0],  // level
            smoothed.smoothed_states[t][1]   // trend
        };
    }
    
    // Compute residuals
    result.residuals.resize(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        result.residuals[t] = data[t] - result.state_estimates[t][0];
    }
    
    // Final state estimates
    level_ = result.state_estimates.back()[0];
    trend_ = result.state_estimates.back()[1];
    
    // Fit statistics
    result.log_likelihood = computeLogLikelihood(data);
    int n_params = 3;  // sigma_obs, sigma_level, sigma_trend
    result.aic = computeAIC(result.log_likelihood, n_params);
    result.bic = computeBIC(result.log_likelihood, n_params, data.size());
    result.converged = true;
    result.iterations = 1;
    
    result.parameters = {
        {"level", level_},
        {"trend", trend_},
        {"sigma_obs", sigma_obs_},
        {"sigma_level", sigma_level_},
        {"sigma_trend", sigma_trend_}
    };
    
    return result;
}

ForecastResult LocalLinearTrendModel::forecast(int horizon,
                                              const std::vector<double>& historical_data,
                                              double confidence) {
    ForecastResult result;
    result.point_forecast.resize(horizon);
    result.lower_bound.resize(horizon);
    result.upper_bound.resize(horizon);
    result.forecast_errors.resize(horizon);
    
    // Initialize from historical data if provided
    double current_level = level_;
    double current_trend = trend_;
    
    if (!historical_data.empty()) {
        FitResult fit_result = fit(historical_data);
        current_level = fit_result.parameters["level"];
        current_trend = fit_result.parameters["trend"];
    }
    
    // Z-score for confidence interval
    double z_score = 1.96;
    if (confidence == 0.99) z_score = 2.576;
    else if (confidence == 0.90) z_score = 1.645;
    
    // Generate forecasts
    for (int h = 0; h < horizon; ++h) {
        // Point forecast with linear trend
        result.point_forecast[h] = current_level + (h + 1) * current_trend;
        
        // Forecast variance
        double var_obs = sigma_obs_ * sigma_obs_;
        double var_level = (h + 1) * sigma_level_ * sigma_level_;
        double var_trend = (h + 1) * (h + 1) * sigma_trend_ * sigma_trend_;
        double forecast_var = var_obs + var_level + var_trend;
        
        result.forecast_errors[h] = std::sqrt(forecast_var);
        
        // Prediction intervals
        result.lower_bound[h] = result.point_forecast[h] - z_score * result.forecast_errors[h];
        result.upper_bound[h] = result.point_forecast[h] + z_score * result.forecast_errors[h];
    }
    
    return result;
}

void LocalLinearTrendModel::setParameters(const json& params) {
    if (params.contains("level")) level_ = params["level"];
    if (params.contains("trend")) trend_ = params["trend"];
    if (params.contains("sigma_obs")) sigma_obs_ = params["sigma_obs"];
    if (params.contains("sigma_level")) sigma_level_ = params["sigma_level"];
    if (params.contains("sigma_trend")) sigma_trend_ = params["sigma_trend"];
}

json LocalLinearTrendModel::getParameters() const {
    return {
        {"level", level_},
        {"trend", trend_},
        {"sigma_obs", sigma_obs_},
        {"sigma_level", sigma_level_},
        {"sigma_trend", sigma_trend_}
    };
}

Eigen::MatrixXd LocalLinearTrendModel::getTransitionMatrix() const {
    // State = [level, trend]'
    Eigen::MatrixXd T(2, 2);
    T << 1, 1,   // level[t] = level[t-1] + trend[t-1]
         0, 1;   // trend[t] = trend[t-1]
    return T;
}

Eigen::MatrixXd LocalLinearTrendModel::getObservationMatrix() const {
    Eigen::MatrixXd Z(1, 2);
    Z << 1, 0;  // Observe only the level
    return Z;
}

Eigen::MatrixXd LocalLinearTrendModel::getStateCovariance() const {
    Eigen::MatrixXd Q(2, 2);
    Q << sigma_level_ * sigma_level_, 0,
         0, sigma_trend_ * sigma_trend_;
    return Q;
}

Eigen::MatrixXd LocalLinearTrendModel::getObservationCovariance() const {
    Eigen::MatrixXd H(1, 1);
    H(0, 0) = sigma_obs_ * sigma_obs_;
    return H;
}