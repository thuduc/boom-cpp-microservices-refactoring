#include "linear_regression.hpp"
#include <Eigen/Dense>
#include <cmath>

FitResult LinearRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // Choose fitting method based on regularization
    if (regularization_lambda_ == 0.0) {
        return fitOLS(X, y);
    } else if (regularization_type_ == "l2") {
        return fitRidge(X, y);
    } else {
        return fitIterative(X, y);
    }
}

FitResult LinearRegression::fitOLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    FitResult result;
    
    // Prepare data
    Eigen::MatrixXd X_work = X;
    Eigen::VectorXd y_work = y;
    
    // Standardize if requested
    if (standardize_) {
        standardizeData(X_work, y_work);
    }
    
    // Add intercept column if needed
    Eigen::MatrixXd X_design = addIntercept(X_work);
    
    // Apply weights if provided
    if (weights_.size() == y.size()) {
        Eigen::VectorXd sqrt_w = weights_.array().sqrt();
        X_design = sqrt_w.asDiagonal() * X_design;
        y_work = sqrt_w.asDiagonal() * y_work;
    }
    
    // Solve normal equations: (X'X)β = X'y
    Eigen::MatrixXd XtX = X_design.transpose() * X_design;
    Eigen::VectorXd Xty = X_design.transpose() * y_work;
    
    // Solve using Cholesky decomposition
    Eigen::VectorXd beta = XtX.ldlt().solve(Xty);
    
    // Extract coefficients and intercept
    if (fit_intercept_) {
        intercept_ = beta(0);
        coefficients_ = beta.tail(beta.size() - 1);
    } else {
        intercept_ = 0.0;
        coefficients_ = beta;
    }
    
    // Unstandardize if needed
    if (standardize_) {
        unstandardizeCoefficients();
    }
    
    // Compute residuals and variance
    Eigen::VectorXd residuals = y - predict(X);
    int n = y.size();
    int p = coefficients_.size() + (fit_intercept_ ? 1 : 0);
    residual_variance_ = residuals.squaredNorm() / (n - p);
    
    // Compute standard errors
    Eigen::MatrixXd cov_matrix = residual_variance_ * XtX.inverse();
    result.standard_errors = cov_matrix.diagonal().array().sqrt();
    
    // Compute p-values (assuming normal distribution)
    result.p_values.resize(p);
    result.confidence_intervals.resize(p);
    
    for (int i = 0; i < p; ++i) {
        double t_stat = beta(i) / result.standard_errors(i);
        // Approximate p-value using normal distribution
        double p_val = 2.0 * (1.0 - std::erf(std::abs(t_stat) / std::sqrt(2.0)) / 2.0);
        result.p_values(i) = p_val;
        
        // 95% confidence intervals
        double margin = 1.96 * result.standard_errors(i);
        result.confidence_intervals[i] = {beta(i) - margin, beta(i) + margin};
    }
    
    // Fill result
    result.coefficients = coefficients_;
    result.intercept = intercept_;
    result.iterations = 1;
    result.converged = true;
    result.log_likelihood = computeLogLikelihood(X, y);
    result.aic = computeAIC(result.log_likelihood, p);
    result.bic = computeBIC(result.log_likelihood, p, n);
    
    return result;
}

FitResult LinearRegression::fitRidge(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    FitResult result;
    
    // Prepare data
    Eigen::MatrixXd X_work = X;
    Eigen::VectorXd y_work = y;
    
    // Standardize if requested
    if (standardize_) {
        standardizeData(X_work, y_work);
    }
    
    // Add intercept column if needed
    Eigen::MatrixXd X_design = addIntercept(X_work);
    
    // Apply weights if provided
    if (weights_.size() == y.size()) {
        Eigen::VectorXd sqrt_w = weights_.array().sqrt();
        X_design = sqrt_w.asDiagonal() * X_design;
        y_work = sqrt_w.asDiagonal() * y_work;
    }
    
    // Ridge regression: (X'X + λI)β = X'y
    Eigen::MatrixXd XtX = X_design.transpose() * X_design;
    Eigen::VectorXd Xty = X_design.transpose() * y_work;
    
    // Add regularization (except for intercept)
    int start_idx = fit_intercept_ ? 1 : 0;
    for (int i = start_idx; i < XtX.rows(); ++i) {
        XtX(i, i) += regularization_lambda_;
    }
    
    // Solve
    Eigen::VectorXd beta = XtX.ldlt().solve(Xty);
    
    // Extract coefficients and intercept
    if (fit_intercept_) {
        intercept_ = beta(0);
        coefficients_ = beta.tail(beta.size() - 1);
    } else {
        intercept_ = 0.0;
        coefficients_ = beta;
    }
    
    // Unstandardize if needed
    if (standardize_) {
        unstandardizeCoefficients();
    }
    
    // Compute residuals and variance
    Eigen::VectorXd residuals = y - predict(X);
    int n = y.size();
    int p = coefficients_.size() + (fit_intercept_ ? 1 : 0);
    residual_variance_ = residuals.squaredNorm() / (n - p);
    
    // Fill result
    result.coefficients = coefficients_;
    result.intercept = intercept_;
    result.iterations = 1;
    result.converged = true;
    result.log_likelihood = computeLogLikelihood(X, y);
    result.aic = computeAIC(result.log_likelihood, p);
    result.bic = computeBIC(result.log_likelihood, p, n);
    
    return result;
}

FitResult LinearRegression::fitIterative(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // For L1 and elastic net, we would implement coordinate descent
    // For now, fall back to OLS
    return fitOLS(X, y);
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& X) const {
    return predictLinear(X);
}

double LinearRegression::computeLogLikelihood(const Eigen::MatrixXd& X, 
                                             const Eigen::VectorXd& y) const {
    Eigen::VectorXd residuals = y - predict(X);
    int n = y.size();
    
    // Normal log-likelihood
    double log_lik = -0.5 * n * std::log(2 * M_PI * residual_variance_);
    log_lik -= 0.5 * residuals.squaredNorm() / residual_variance_;
    
    return log_lik;
}

json LinearRegression::predictionIntervals(const Eigen::MatrixXd& X, double alpha) const {
    json intervals;
    
    // Compute predictions
    Eigen::VectorXd predictions = predict(X);
    
    // Standard error of predictions
    double z_score = 1.96;  // For 95% CI (approximation)
    if (alpha == 0.01) z_score = 2.576;
    else if (alpha == 0.10) z_score = 1.645;
    
    double se = std::sqrt(residual_variance_);
    
    std::vector<double> lower(predictions.size());
    std::vector<double> upper(predictions.size());
    
    for (int i = 0; i < predictions.size(); ++i) {
        lower[i] = predictions(i) - z_score * se;
        upper[i] = predictions(i) + z_score * se;
    }
    
    intervals["lower"] = lower;
    intervals["upper"] = upper;
    intervals["alpha"] = alpha;
    
    return intervals;
}