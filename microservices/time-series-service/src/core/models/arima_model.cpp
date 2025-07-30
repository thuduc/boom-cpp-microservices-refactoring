#include "arima_model.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

ARIMAModel::ARIMAModel(int p, int d, int q) : p_(p), d_(d), q_(q), intercept_(0.0), sigma2_(1.0) {
    ar_coeffs_.resize(p_, 0.0);
    ma_coeffs_.resize(q_, 0.0);
}

void ARIMAModel::setOrders(int p, int d, int q) {
    p_ = p;
    d_ = d;
    q_ = q;
    ar_coeffs_.resize(p_, 0.0);
    ma_coeffs_.resize(q_, 0.0);
}

std::vector<double> ARIMAModel::difference(const std::vector<double>& data, int d) {
    if (d == 0) return data;
    
    std::vector<double> result = data;
    for (int i = 0; i < d; ++i) {
        std::vector<double> temp;
        for (size_t j = 1; j < result.size(); ++j) {
            temp.push_back(result[j] - result[j-1]);
        }
        result = temp;
    }
    return result;
}

std::vector<double> ARIMAModel::undifference(const std::vector<double>& data, 
                                            const std::vector<double>& original, int d) {
    if (d == 0) return data;
    
    std::vector<double> result = data;
    
    // Reverse the differencing process
    for (int i = d - 1; i >= 0; --i) {
        std::vector<double> temp;
        
        // Get the appropriate starting value from original series
        double last_val = original[d - i - 1];
        temp.push_back(last_val);
        
        for (double val : result) {
            last_val += val;
            temp.push_back(last_val);
        }
        
        result = std::vector<double>(temp.begin() + 1, temp.end());
    }
    
    return result;
}

void ARIMAModel::estimateParameters(const std::vector<double>& data) {
    // Simplified parameter estimation using method of moments
    // In practice, use maximum likelihood or conditional sum of squares
    
    size_t n = data.size();
    
    // Estimate mean
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    
    // Estimate autocorrelations
    std::vector<double> acf(std::max(p_, q_) + 1, 0.0);
    double c0 = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        c0 += (data[i] - mean) * (data[i] - mean);
    }
    c0 /= n;
    
    for (size_t k = 1; k <= acf.size() - 1; ++k) {
        double ck = 0.0;
        for (size_t i = k; i < n; ++i) {
            ck += (data[i] - mean) * (data[i-k] - mean);
        }
        acf[k] = ck / (n * c0);
    }
    
    // Simple AR coefficient estimation (Yule-Walker)
    if (p_ > 0) {
        Eigen::MatrixXd R(p_, p_);
        Eigen::VectorXd r(p_);
        
        for (int i = 0; i < p_; ++i) {
            r(i) = acf[i + 1];
            for (int j = 0; j < p_; ++j) {
                R(i, j) = acf[std::abs(i - j)];
            }
        }
        
        Eigen::VectorXd phi = R.ldlt().solve(r);
        for (int i = 0; i < p_; ++i) {
            ar_coeffs_[i] = phi(i);
        }
    }
    
    // MA coefficients initialization (simplified)
    if (q_ > 0) {
        for (int i = 0; i < q_; ++i) {
            ma_coeffs_[i] = 0.1 * (1.0 - i / static_cast<double>(q_));
        }
    }
    
    intercept_ = mean * (1.0 - std::accumulate(ar_coeffs_.begin(), ar_coeffs_.end(), 0.0));
    sigma2_ = c0;
}

double ARIMAModel::computeLogLikelihood(const std::vector<double>& data) const {
    size_t n = data.size();
    double log_lik = -0.5 * n * std::log(2 * M_PI * sigma2_);
    
    // Compute residuals
    std::vector<double> residuals(n, 0.0);
    
    for (size_t t = std::max(p_, q_); t < n; ++t) {
        double pred = intercept_;
        
        // AR part
        for (int i = 0; i < p_; ++i) {
            if (t > i) {
                pred += ar_coeffs_[i] * data[t - i - 1];
            }
        }
        
        // MA part (using previous residuals)
        for (int i = 0; i < q_; ++i) {
            if (t > i) {
                pred += ma_coeffs_[i] * residuals[t - i - 1];
            }
        }
        
        residuals[t] = data[t] - pred;
        log_lik -= 0.5 * residuals[t] * residuals[t] / sigma2_;
    }
    
    return log_lik;
}

FitResult ARIMAModel::fit(const std::vector<double>& data, 
                         const std::vector<double>& timestamps) {
    FitResult result;
    
    // Apply differencing
    std::vector<double> working_data = difference(data, d_);
    
    // Estimate parameters
    estimateParameters(working_data);
    
    // Compute fit statistics
    result.log_likelihood = computeLogLikelihood(working_data);
    int n_params = p_ + q_ + 1 + (d_ == 0 ? 1 : 0);  // AR + MA + sigma2 + intercept
    result.aic = computeAIC(result.log_likelihood, n_params);
    result.bic = computeBIC(result.log_likelihood, n_params, working_data.size());
    result.converged = true;
    result.iterations = 1;  // Simplified - actual implementation would iterate
    
    // Store parameters
    result.parameters = {
        {"p", p_},
        {"d", d_},
        {"q", q_},
        {"ar_coefficients", ar_coeffs_},
        {"ma_coefficients", ma_coeffs_},
        {"intercept", intercept_},
        {"sigma2", sigma2_}
    };
    
    // Compute residuals on original scale
    std::vector<double> fitted_diff(working_data.size());
    for (size_t t = 0; t < working_data.size(); ++t) {
        double pred = intercept_;
        
        for (int i = 0; i < p_; ++i) {
            if (t > i) {
                pred += ar_coeffs_[i] * working_data[t - i - 1];
            }
        }
        
        fitted_diff[t] = pred;
    }
    
    // Undifference fitted values
    if (d_ > 0) {
        std::vector<double> fitted = undifference(fitted_diff, data, d_);
        result.residuals.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.residuals[i] = data[i] - fitted[i];
        }
    } else {
        result.residuals.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result.residuals[i] = data[i] - fitted_diff[i];
        }
    }
    
    return result;
}

ForecastResult ARIMAModel::forecast(int horizon,
                                   const std::vector<double>& historical_data,
                                   double confidence) {
    ForecastResult result;
    result.point_forecast.resize(horizon);
    result.lower_bound.resize(horizon);
    result.upper_bound.resize(horizon);
    result.forecast_errors.resize(horizon);
    
    // Use historical data to initialize if provided
    std::vector<double> working_data;
    if (!historical_data.empty()) {
        working_data = difference(historical_data, d_);
    }
    
    // Z-score for confidence interval
    double z_score = 1.96;  // 95% confidence
    if (confidence == 0.99) z_score = 2.576;
    else if (confidence == 0.90) z_score = 1.645;
    
    // Generate forecasts
    for (int h = 0; h < horizon; ++h) {
        double forecast = intercept_;
        
        // AR component
        for (int i = 0; i < p_; ++i) {
            if (h > i) {
                // Use previous forecasts
                forecast += ar_coeffs_[i] * result.point_forecast[h - i - 1];
            } else if (!working_data.empty() && working_data.size() > i - h) {
                // Use historical data
                forecast += ar_coeffs_[i] * working_data[working_data.size() - 1 - (i - h)];
            }
        }
        
        // MA component (assuming zero future shocks)
        // In practice, would use estimated residuals
        
        result.point_forecast[h] = forecast;
        
        // Forecast error variance increases with horizon
        double forecast_var = sigma2_ * (1.0 + h * 0.1);  // Simplified
        result.forecast_errors[h] = std::sqrt(forecast_var);
        
        result.lower_bound[h] = forecast - z_score * result.forecast_errors[h];
        result.upper_bound[h] = forecast + z_score * result.forecast_errors[h];
    }
    
    // Undifference forecasts if needed
    if (d_ > 0 && !historical_data.empty()) {
        // Get last d values from historical data for undifferencing
        std::vector<double> last_values(historical_data.end() - d_, historical_data.end());
        
        for (int i = 0; i < d_; ++i) {
            double cumsum = last_values[d_ - 1 - i];
            for (int h = 0; h < horizon; ++h) {
                result.point_forecast[h] += cumsum;
                result.lower_bound[h] += cumsum;
                result.upper_bound[h] += cumsum;
            }
        }
    }
    
    return result;
}

void ARIMAModel::setParameters(const json& params) {
    if (params.contains("p")) p_ = params["p"];
    if (params.contains("d")) d_ = params["d"];
    if (params.contains("q")) q_ = params["q"];
    
    if (params.contains("ar_coefficients")) {
        ar_coeffs_ = params["ar_coefficients"].get<std::vector<double>>();
    }
    if (params.contains("ma_coefficients")) {
        ma_coeffs_ = params["ma_coefficients"].get<std::vector<double>>();
    }
    if (params.contains("intercept")) intercept_ = params["intercept"];
    if (params.contains("sigma2")) sigma2_ = params["sigma2"];
}

json ARIMAModel::getParameters() const {
    return {
        {"p", p_},
        {"d", d_},
        {"q", q_},
        {"ar_coefficients", ar_coeffs_},
        {"ma_coefficients", ma_coeffs_},
        {"intercept", intercept_},
        {"sigma2", sigma2_}
    };
}

Eigen::MatrixXd ARIMAModel::getTransitionMatrix() const {
    // State space representation of ARIMA
    int state_dim = std::max(p_, q_ + 1);
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(state_dim, state_dim);
    
    // First row contains AR coefficients
    for (int i = 0; i < p_; ++i) {
        T(0, i) = ar_coeffs_[i];
    }
    
    // Companion matrix structure
    for (int i = 1; i < state_dim; ++i) {
        T(i, i - 1) = 1.0;
    }
    
    return T;
}

Eigen::MatrixXd ARIMAModel::getObservationMatrix() const {
    int state_dim = std::max(p_, q_ + 1);
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1, state_dim);
    Z(0, 0) = 1.0;
    return Z;
}

Eigen::MatrixXd ARIMAModel::getStateCovariance() const {
    int state_dim = std::max(p_, q_ + 1);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(state_dim, state_dim);
    
    // Innovation enters only in first state
    Q(0, 0) = sigma2_;
    
    // MA coefficients affect covariance structure
    for (int i = 0; i < q_; ++i) {
        Q(0, i + 1) = ma_coeffs_[i] * sigma2_;
        Q(i + 1, 0) = ma_coeffs_[i] * sigma2_;
    }
    
    return Q;
}

Eigen::MatrixXd ARIMAModel::getObservationCovariance() const {
    Eigen::MatrixXd H(1, 1);
    H(0, 0) = 0.0;  // No observation noise in basic ARIMA
    return H;
}