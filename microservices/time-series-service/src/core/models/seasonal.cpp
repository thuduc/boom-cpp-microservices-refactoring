#include "seasonal.hpp"
#include "../filters/kalman_filter.hpp"
#include <numeric>
#include <cmath>

SeasonalModel::SeasonalModel(int period) 
    : period_(period), level_(0.0), trend_(0.0), 
      sigma_obs_(1.0), sigma_level_(0.1), sigma_trend_(0.01), sigma_seasonal_(0.1) {
    seasonal_.resize(period_, 0.0);
}

void SeasonalModel::estimateParametersEM(const std::vector<double>& data) {
    // Simplified initialization
    level_ = std::accumulate(data.begin(), data.begin() + std::min(period_, (int)data.size()), 0.0) / 
             std::min(period_, (int)data.size());
    
    if (data.size() > period_) {
        trend_ = (data[data.size()-1] - data[0]) / (data.size() - 1);
    }
    
    // Initialize seasonal pattern
    for (int i = 0; i < period_ && i < data.size(); ++i) {
        seasonal_[i] = data[i] - level_ - i * trend_;
    }
    
    // Normalize seasonal components
    double seasonal_mean = std::accumulate(seasonal_.begin(), seasonal_.end(), 0.0) / period_;
    for (double& s : seasonal_) {
        s -= seasonal_mean;
    }
    
    // Simple variance estimates
    double var = 0.0;
    for (size_t i = 1; i < data.size(); ++i) {
        var += (data[i] - data[i-1]) * (data[i] - data[i-1]);
    }
    var /= (data.size() - 1);
    
    sigma_obs_ = std::sqrt(var * 0.5);
    sigma_level_ = std::sqrt(var * 0.2);
    sigma_trend_ = std::sqrt(var * 0.1);
    sigma_seasonal_ = std::sqrt(var * 0.2);
}

double SeasonalModel::computeLogLikelihood(const std::vector<double>& data) const {
    // Simplified likelihood calculation
    double log_lik = 0.0;
    
    for (size_t t = 0; t < data.size(); ++t) {
        double pred = level_ + t * trend_ + seasonal_[t % period_];
        double error = data[t] - pred;
        log_lik -= 0.5 * (std::log(2 * M_PI * sigma_obs_ * sigma_obs_) + 
                         error * error / (sigma_obs_ * sigma_obs_));
    }
    
    return log_lik;
}

FitResult SeasonalModel::fit(const std::vector<double>& data, 
                            const std::vector<double>& timestamps) {
    FitResult result;
    
    // Estimate parameters
    estimateParametersEM(data);
    
    // Extract components
    result.state_estimates.resize(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        double level_t = level_ + t * trend_;
        double seasonal_t = seasonal_[t % period_];
        result.state_estimates[t] = {level_t, trend_, seasonal_t};
    }
    
    // Compute residuals
    result.residuals.resize(data.size());
    for (size_t t = 0; t < data.size(); ++t) {
        double pred = result.state_estimates[t][0] + result.state_estimates[t][2];
        result.residuals[t] = data[t] - pred;
    }
    
    // Fit statistics
    result.log_likelihood = computeLogLikelihood(data);
    int n_params = 4 + period_;  // 4 variance parameters + seasonal components
    result.aic = computeAIC(result.log_likelihood, n_params);
    result.bic = computeBIC(result.log_likelihood, n_params, data.size());
    result.converged = true;
    result.iterations = 1;
    
    result.parameters = {
        {"level", level_},
        {"trend", trend_},
        {"seasonal", seasonal_},
        {"period", period_},
        {"sigma_obs", sigma_obs_},
        {"sigma_level", sigma_level_},
        {"sigma_trend", sigma_trend_},
        {"sigma_seasonal", sigma_seasonal_}
    };
    
    return result;
}

ForecastResult SeasonalModel::forecast(int horizon,
                                      const std::vector<double>& historical_data,
                                      double confidence) {
    ForecastResult result;
    result.point_forecast.resize(horizon);
    result.lower_bound.resize(horizon);
    result.upper_bound.resize(horizon);
    result.forecast_errors.resize(horizon);
    
    // Update parameters if historical data provided
    double current_level = level_;
    double current_trend = trend_;
    std::vector<double> current_seasonal = seasonal_;
    
    if (!historical_data.empty()) {
        FitResult fit_result = fit(historical_data);
        current_level = fit_result.parameters["level"];
        current_trend = fit_result.parameters["trend"];
        current_seasonal = fit_result.parameters["seasonal"].get<std::vector<double>>();
        
        // Adjust level for end of historical data
        current_level += historical_data.size() * current_trend;
    }
    
    // Z-score for confidence interval
    double z_score = 1.96;
    if (confidence == 0.99) z_score = 2.576;
    else if (confidence == 0.90) z_score = 1.645;
    
    // Generate forecasts
    for (int h = 0; h < horizon; ++h) {
        // Point forecast
        int seasonal_idx = h % period_;
        result.point_forecast[h] = current_level + (h + 1) * current_trend + 
                                  current_seasonal[seasonal_idx];
        
        // Forecast variance (simplified)
        double forecast_var = sigma_obs_ * sigma_obs_ + 
                             (h + 1) * sigma_level_ * sigma_level_ +
                             (h + 1) * (h + 1) * sigma_trend_ * sigma_trend_ +
                             sigma_seasonal_ * sigma_seasonal_;
        
        result.forecast_errors[h] = std::sqrt(forecast_var);
        
        // Prediction intervals
        result.lower_bound[h] = result.point_forecast[h] - z_score * result.forecast_errors[h];
        result.upper_bound[h] = result.point_forecast[h] + z_score * result.forecast_errors[h];
    }
    
    return result;
}

void SeasonalModel::setParameters(const json& params) {
    if (params.contains("level")) level_ = params["level"];
    if (params.contains("trend")) trend_ = params["trend"];
    if (params.contains("seasonal")) {
        seasonal_ = params["seasonal"].get<std::vector<double>>();
        period_ = seasonal_.size();
    }
    if (params.contains("period")) period_ = params["period"];
    if (params.contains("sigma_obs")) sigma_obs_ = params["sigma_obs"];
    if (params.contains("sigma_level")) sigma_level_ = params["sigma_level"];
    if (params.contains("sigma_trend")) sigma_trend_ = params["sigma_trend"];
    if (params.contains("sigma_seasonal")) sigma_seasonal_ = params["sigma_seasonal"];
}

json SeasonalModel::getParameters() const {
    return {
        {"level", level_},
        {"trend", trend_},
        {"seasonal", seasonal_},
        {"period", period_},
        {"sigma_obs", sigma_obs_},
        {"sigma_level", sigma_level_},
        {"sigma_trend", sigma_trend_},
        {"sigma_seasonal", sigma_seasonal_}
    };
}

Eigen::MatrixXd SeasonalModel::getTransitionMatrix() const {
    // State = [level, trend, seasonal_1, ..., seasonal_{s-1}]'
    int state_dim = 2 + period_ - 1;
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(state_dim, state_dim);
    
    // Level and trend dynamics
    T(0, 0) = 1;  // level
    T(0, 1) = 1;  // trend added to level
    T(1, 1) = 1;  // trend persists
    
    // Seasonal dynamics (circular shift)
    if (period_ > 1) {
        // The new seasonal_1 is the negative sum of all other seasonals
        for (int i = 2; i < state_dim; ++i) {
            T(2, i) = -1;
        }
        
        // Shift seasonal components
        for (int i = 3; i < state_dim; ++i) {
            T(i, i - 1) = 1;
        }
    }
    
    return T;
}

Eigen::MatrixXd SeasonalModel::getObservationMatrix() const {
    int state_dim = 2 + period_ - 1;
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1, state_dim);
    Z(0, 0) = 1;  // level
    if (period_ > 1) {
        Z(0, 2) = 1;  // current seasonal
    }
    return Z;
}

Eigen::MatrixXd SeasonalModel::getStateCovariance() const {
    int state_dim = 2 + period_ - 1;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    
    Q(0, 0) = sigma_level_ * sigma_level_;
    Q(1, 1) = sigma_trend_ * sigma_trend_;
    
    // Seasonal components
    for (int i = 2; i < state_dim; ++i) {
        Q(i, i) = sigma_seasonal_ * sigma_seasonal_;
    }
    
    return Q;
}

Eigen::MatrixXd SeasonalModel::getObservationCovariance() const {
    Eigen::MatrixXd H(1, 1);
    H(0, 0) = sigma_obs_ * sigma_obs_;
    return H;
}