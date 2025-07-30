#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "../glm_base.hpp"

class LinearRegression : public GLMBase {
public:
    LinearRegression() = default;
    
    // Fitting - uses analytical solution or iterative methods
    FitResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    
    // Prediction
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    
    // Prediction intervals (for linear regression)
    json predictionIntervals(const Eigen::MatrixXd& X, double alpha = 0.05) const override;
    
protected:
    // Link functions for linear regression (identity)
    double linkFunction(double linear_pred) const override { return linear_pred; }
    double linkDerivative(double linear_pred) const override { return 1.0; }
    double inverseLinkFunction(double eta) const override { return eta; }
    
    // Log likelihood for normal distribution
    double computeLogLikelihood(const Eigen::MatrixXd& X, 
                               const Eigen::VectorXd& y) const override;
    
private:
    double residual_variance_ = 1.0;  // Estimated from residuals
    
    // Analytical solution for OLS
    FitResult fitOLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Ridge regression solution
    FitResult fitRidge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Iterative solution for L1/Elastic net
    FitResult fitIterative(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

#endif // LINEAR_REGRESSION_HPP