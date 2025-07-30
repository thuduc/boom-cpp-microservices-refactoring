#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include "../models/time_series_model.hpp"
#include <Eigen/Dense>
#include <vector>

struct FilterResult {
    std::vector<Eigen::VectorXd> filtered_states;
    std::vector<Eigen::MatrixXd> filtered_variances;
    std::vector<double> predictions;
    std::vector<double> innovations;
};

struct SmoothResult {
    std::vector<Eigen::VectorXd> smoothed_states;
    std::vector<Eigen::MatrixXd> smoothed_variances;
};

class KalmanFilter {
public:
    // Forward filtering
    FilterResult filter(const TimeSeriesModel* model,
                       const std::vector<double>& observations);
    
    // Backward smoothing
    SmoothResult smooth(const TimeSeriesModel* model,
                       const std::vector<double>& observations);
    
private:
    // Initialize state and covariance
    void initializeState(const TimeSeriesModel* model,
                        Eigen::VectorXd& state,
                        Eigen::MatrixXd& covariance);
};

#endif // KALMAN_FILTER_HPP