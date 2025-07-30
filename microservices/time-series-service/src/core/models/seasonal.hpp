#ifndef SEASONAL_MODEL_HPP
#define SEASONAL_MODEL_HPP

#include "time_series_model.hpp"

class SeasonalModel : public TimeSeriesModel {
public:
    SeasonalModel(int period = 12);
    
    void setPeriod(int period) { period_ = period; }
    
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
    int period_;                    // Seasonal period
    double level_;                  // Current level
    double trend_;                  // Current trend
    std::vector<double> seasonal_;  // Seasonal components
    
    double sigma_obs_;              // Observation noise
    double sigma_level_;            // Level noise
    double sigma_trend_;            // Trend noise
    double sigma_seasonal_;         // Seasonal noise
    
    void estimateParametersEM(const std::vector<double>& data);
    double computeLogLikelihood(const std::vector<double>& data) const;
};

#endif // SEASONAL_MODEL_HPP