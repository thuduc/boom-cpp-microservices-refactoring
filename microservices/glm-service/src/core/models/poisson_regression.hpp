#ifndef POISSON_REGRESSION_HPP
#define POISSON_REGRESSION_HPP

#include "../glm_base.hpp"

class PoissonRegression : public GLMBase {
public:
    PoissonRegression() = default;
    
    // Fitting - uses iteratively reweighted least squares (IRLS)
    FitResult fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    
    // Prediction
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    
protected:
    // Log link functions
    double linkFunction(double mu) const override;
    double linkDerivative(double mu) const override;
    double inverseLinkFunction(double eta) const override;
    
    // Log likelihood for Poisson distribution
    double computeLogLikelihood(const Eigen::MatrixXd& X, 
                               const Eigen::VectorXd& y) const override;
    
private:
    // IRLS algorithm implementation
    FitResult fitIRLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

#endif // POISSON_REGRESSION_HPP