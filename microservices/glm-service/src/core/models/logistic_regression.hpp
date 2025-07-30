#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "../glm_base.hpp"

class LogisticRegression : public GLMBase {
public:
    LogisticRegression() = default;
    
    // Fitting - uses iteratively reweighted least squares (IRLS)
    FitResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    
    // Prediction
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    Eigen::VectorXd predictProbability(const Eigen::MatrixXd& X) const override;
    
protected:
    // Logistic link functions
    double linkFunction(double linear_pred) const override;
    double linkDerivative(double linear_pred) const override;
    double inverseLinkFunction(double eta) const override;
    
    // Log likelihood for binomial distribution
    double computeLogLikelihood(const Eigen::MatrixXd& X, 
                               const Eigen::VectorXd& y) const override;
    
private:
    // IRLS algorithm implementation
    FitResult fitIRLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    
    // Helper for numerical stability
    double sigmoid(double x) const;
};

#endif // LOGISTIC_REGRESSION_HPP