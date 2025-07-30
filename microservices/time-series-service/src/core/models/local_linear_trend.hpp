#ifndef LOCAL_LINEAR_TREND_HPP
#define LOCAL_LINEAR_TREND_HPP

#include "time_series_model.hpp"

class LocalLinearTrendModel : public TimeSeriesModel {
public:
    LocalLinearTrendModel();
    
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
    double level_;           // Current level estimate
    double trend_;           // Current trend estimate
    double sigma_obs_;       // Observation noise
    double sigma_level_;     // Level noise
    double sigma_trend_;     // Trend noise
    
    void estimateParametersEM(const std::vector<double>& data);
    double computeLogLikelihood(const std::vector<double>& data) const;
};

#endif // LOCAL_LINEAR_TREND_HPP