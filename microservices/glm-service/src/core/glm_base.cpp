#include "glm_base.hpp"
#include <cmath>

Eigen::VectorXd GLMBase::predictLinear(const Eigen::MatrixXd& X) const {
    if (fit_intercept_) {
        return X * coefficients_ + Eigen::VectorXd::Constant(X.rows(), intercept_);
    } else {
        return X * coefficients_;
    }
}

void GLMBase::setRegularization(double lambda, const std::string& type) {
    regularization_lambda_ = lambda;
    regularization_type_ = type;
    if (type != "l1" && type != "l2" && type != "elastic") {
        regularization_type_ = "l2";  // Default to L2
    }
}

void GLMBase::setCoefficients(const Eigen::VectorXd& coef, double intercept) {
    coefficients_ = coef;
    intercept_ = intercept;
}

json GLMBase::predictionIntervals(const Eigen::MatrixXd& X, double alpha) const {
    // Default implementation returns empty intervals
    // Specific models should override this
    json intervals;
    intervals["lower"] = std::vector<double>(X.rows(), 0.0);
    intervals["upper"] = std::vector<double>(X.rows(), 0.0);
    intervals["alpha"] = alpha;
    return intervals;
}

Eigen::MatrixXd GLMBase::addIntercept(const Eigen::MatrixXd& X) const {
    if (!fit_intercept_) {
        return X;
    }
    
    Eigen::MatrixXd X_with_intercept(X.rows(), X.cols() + 1);
    X_with_intercept.col(0) = Eigen::VectorXd::Ones(X.rows());
    X_with_intercept.rightCols(X.cols()) = X;
    return X_with_intercept;
}

void GLMBase::standardizeData(Eigen::MatrixXd& X, Eigen::VectorXd& y) {
    if (!standardize_) return;
    
    int n = X.rows();
    int p = X.cols();
    
    // Compute mean and std for X
    mean_ = X.colwise().mean();
    std_ = Eigen::VectorXd(p);
    
    for (int j = 0; j < p; ++j) {
        double variance = (X.col(j).array() - mean_(j)).square().sum() / (n - 1);
        std_(j) = std::sqrt(variance);
        if (std_(j) < 1e-10) std_(j) = 1.0;  // Avoid division by zero
        
        // Standardize column
        X.col(j) = (X.col(j).array() - mean_(j)) / std_(j);
    }
}

void GLMBase::unstandardizeCoefficients() {
    if (!standardize_) return;
    
    // Transform coefficients back to original scale
    for (int j = 0; j < coefficients_.size(); ++j) {
        coefficients_(j) /= std_(j);
        intercept_ -= coefficients_(j) * mean_(j);
    }
}

double GLMBase::computeAIC(double log_likelihood, int n_params) const {
    return -2.0 * log_likelihood + 2.0 * n_params;
}

double GLMBase::computeBIC(double log_likelihood, int n_params, int n_obs) const {
    return -2.0 * log_likelihood + n_params * std::log(n_obs);
}