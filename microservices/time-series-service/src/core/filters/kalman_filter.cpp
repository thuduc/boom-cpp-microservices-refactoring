#include "kalman_filter.hpp"
#include <cmath>

void KalmanFilter::initializeState(const TimeSeriesModel* model,
                                  Eigen::VectorXd& state,
                                  Eigen::MatrixXd& covariance) {
    auto T = model->getTransitionMatrix();
    auto Q = model->getStateCovariance();
    
    int state_dim = T.rows();
    state = Eigen::VectorXd::Zero(state_dim);
    
    // Initialize covariance using steady-state solution
    // P = Q + T*P*T' => vec(P) = (I - T⊗T)^(-1) vec(Q)
    // Simplified: use large diagonal matrix
    covariance = Eigen::MatrixXd::Identity(state_dim, state_dim) * 1000.0;
}

FilterResult KalmanFilter::filter(const TimeSeriesModel* model,
                                 const std::vector<double>& observations) {
    FilterResult result;
    
    // Get state space matrices
    auto T = model->getTransitionMatrix();
    auto Z = model->getObservationMatrix();
    auto Q = model->getStateCovariance();
    auto H = model->getObservationCovariance();
    
    int state_dim = T.rows();
    int obs_dim = Z.rows();
    
    // Initialize state and covariance
    Eigen::VectorXd state;
    Eigen::MatrixXd P;
    initializeState(model, state, P);
    
    // Reserve space for results
    result.filtered_states.reserve(observations.size());
    result.filtered_variances.reserve(observations.size());
    result.predictions.reserve(observations.size());
    result.innovations.reserve(observations.size());
    
    // Kalman filter recursion
    for (size_t t = 0; t < observations.size(); ++t) {
        // Prediction step
        Eigen::VectorXd state_pred = T * state;
        Eigen::MatrixXd P_pred = T * P * T.transpose() + Q;
        
        // Observation prediction
        Eigen::VectorXd y_pred = Z * state_pred;
        double prediction = y_pred(0);
        result.predictions.push_back(prediction);
        
        // Innovation
        double innovation = observations[t] - prediction;
        result.innovations.push_back(innovation);
        
        // Innovation variance
        Eigen::MatrixXd F = Z * P_pred * Z.transpose() + H;
        double F_inv = 1.0 / F(0, 0);
        
        // Kalman gain
        Eigen::VectorXd K = P_pred * Z.transpose() * F_inv;
        
        // Update step
        state = state_pred + K * innovation;
        P = P_pred - K * F * K.transpose();
        
        // Store results
        result.filtered_states.push_back(state);
        result.filtered_variances.push_back(P);
    }
    
    return result;
}

SmoothResult KalmanFilter::smooth(const TimeSeriesModel* model,
                                 const std::vector<double>& observations) {
    SmoothResult result;
    
    // First run forward filter
    auto filter_result = filter(model, observations);
    
    // Get state space matrices
    auto T = model->getTransitionMatrix();
    auto Q = model->getStateCovariance();
    
    int n = observations.size();
    result.smoothed_states.resize(n);
    result.smoothed_variances.resize(n);
    
    // Initialize with final filtered values
    result.smoothed_states[n-1] = filter_result.filtered_states[n-1];
    result.smoothed_variances[n-1] = filter_result.filtered_variances[n-1];
    
    // Backward recursion
    for (int t = n - 2; t >= 0; --t) {
        // Prediction for time t+1
        Eigen::VectorXd state_pred = T * filter_result.filtered_states[t];
        Eigen::MatrixXd P_pred = T * filter_result.filtered_variances[t] * T.transpose() + Q;
        
        // Smoother gain
        Eigen::MatrixXd J = filter_result.filtered_variances[t] * T.transpose() * P_pred.inverse();
        
        // Smoothed estimates
        result.smoothed_states[t] = filter_result.filtered_states[t] + 
            J * (result.smoothed_states[t+1] - state_pred);
        
        result.smoothed_variances[t] = filter_result.filtered_variances[t] + 
            J * (result.smoothed_variances[t+1] - P_pred) * J.transpose();
    }
    
    return result;
}