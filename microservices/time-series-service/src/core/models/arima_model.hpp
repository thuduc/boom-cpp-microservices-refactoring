#ifndef ARIMA_MODEL_HPP
#define ARIMA_MODEL_HPP

#include "time_series_model.hpp"
#include <vector>

class ARIMAModel : public TimeSeriesModel {
public:
    ARIMAModel(int p = 1, int d = 0, int q = 1);
    
    // Set ARIMA orders
    void setOrders(int p, int d, int q);
    
    // TimeSeriesModel interface
    FitResult fit(const std::vector<double>& data, 
                 const std::vector<double>& timestamps = {}) override;
    
    ForecastResult forecast(int horizon,
                           const std::vector<double>& historical_data = {},
                           double confidence = 0.95) override;
    
    void setParameters(const json& params) override;
    json getParameters() const override;
    
    // State space representation
    Eigen::MatrixXd getTransitionMatrix() const override;
    Eigen::MatrixXd getObservationMatrix() const override;
    Eigen::MatrixXd getStateCovariance() const override;
    Eigen::MatrixXd getObservationCovariance() const override;
    
private:
    int p_;  // AR order
    int d_;  // Differencing order
    int q_;  // MA order
    
    std::vector<double> ar_coeffs_;
    std::vector<double> ma_coeffs_;
    double intercept_;
    double sigma2_;  // Innovation variance
    
    // Helper methods
    std::vector<double> difference(const std::vector<double>& data, int d);
    std::vector<double> undifference(const std::vector<double>& data, 
                                    const std::vector<double>& original, int d);
    void estimateParameters(const std::vector<double>& data);
    double computeLogLikelihood(const std::vector<double>& data) const;
};

#endif // ARIMA_MODEL_HPP