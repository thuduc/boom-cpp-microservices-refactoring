#ifndef LOCAL_LEVEL_HPP
#define LOCAL_LEVEL_HPP

#include "time_series_model.hpp"

class LocalLevelModel : public TimeSeriesModel {
public:
    LocalLevelModel();
    
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
    double sigma_obs_;       // Observation noise standard deviation
    double sigma_level_;     // Level noise standard deviation
    
    // Kalman filter for parameter estimation
    void estimateParametersEM(const std::vector<double>& data);
    double computeLogLikelihood(const std::vector<double>& data) const;
};

#endif // LOCAL_LEVEL_HPP