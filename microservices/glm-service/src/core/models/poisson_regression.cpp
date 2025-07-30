#include "poisson_regression.hpp"
#include <cmath>
#include <algorithm>

double PoissonRegression::linkFunction(double mu) const {
    // Log link
    return std::log(std::max(1e-10, mu));
}

double PoissonRegression::linkDerivative(double mu) const {
    // Derivative of log link: 1/mu
    return 1.0 / std::max(1e-10, mu);
}

double PoissonRegression::inverseLinkFunction(double eta) const {
    // Inverse log link (exp)
    return std::exp(eta);
}

FitResult PoissonRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    return fitIRLS(X, y);
}

FitResult PoissonRegression::fitIRLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    FitResult result;
    
    // Check that y contains non-negative integers
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) < 0 || std::floor(y(i)) != y(i)) {
            throw std::runtime_error("Poisson regression requires non-negative integer outcomes");
        }
    }
    
    // Prepare data
    Eigen::MatrixXd X_work = X;
    Eigen::VectorXd y_work = y;
    
    // Standardize if requested
    if (standardize_) {
        standardizeData(X_work, y_work);
    }
    
    // Add intercept column if needed
    Eigen::MatrixXd X_design = addIntercept(X_work);
    
    // Initialize coefficients
    int p = X_design.cols();
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    
    // Initialize with log of mean as intercept
    if (fit_intercept_) {
        double mean_y = y_work.mean();
        beta(0) = std::log(std::max(0.1, mean_y));
    }
    
    // IRLS algorithm
    const int max_iter = 100;
    const double tol = 1e-6;
    bool converged = false;
    int iter = 0;
    
    for (iter = 0; iter < max_iter; ++iter) {
        // Compute linear predictor
        Eigen::VectorXd eta = X_design * beta;
        
        // Compute mean function (exp(eta))
        Eigen::VectorXd mu(eta.size());
        for (int i = 0; i < eta.size(); ++i) {
            mu(i) = std::exp(std::min(20.0, eta(i)));  // Prevent overflow
        }
        
        // Compute weights and working response
        Eigen::VectorXd W(mu.size());
        Eigen::VectorXd z(mu.size());
        
        for (int i = 0; i < mu.size(); ++i) {
            double mui = std::max(1e-10, mu(i));
            W(i) = mui;  // For Poisson with log link, W = mu
            z(i) = eta(i) + (y_work(i) - mui) / mui;
            
            // Apply observation weights if provided
            if (weights_.size() == y.size()) {
                W(i) *= weights_(i);
            }
        }
        
        // Weighted least squares update
        Eigen::MatrixXd XtWX = X_design.transpose() * W.asDiagonal() * X_design;
        Eigen::VectorXd XtWz = X_design.transpose() * W.asDiagonal() * z;
        
        // Add regularization if needed
        if (regularization_lambda_ > 0) {
            int start_idx = fit_intercept_ ? 1 : 0;
            for (int i = start_idx; i < p; ++i) {
                XtWX(i, i) += regularization_lambda_;
            }
        }
        
        // Solve for new beta
        Eigen::VectorXd beta_new = XtWX.ldlt().solve(XtWz);
        
        // Check convergence
        double change = (beta_new - beta).norm();
        beta = beta_new;
        
        if (change < tol) {
            converged = true;
            break;
        }
    }
    
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
    
    // Compute final statistics
    result.coefficients = coefficients_;
    result.intercept = intercept_;
    result.iterations = iter + 1;
    result.converged = converged;
    result.log_likelihood = computeLogLikelihood(X, y);
    
    int n = y.size();
    result.aic = computeAIC(result.log_likelihood, p);
    result.bic = computeBIC(result.log_likelihood, p, n);
    
    // Compute standard errors
    Eigen::VectorXd eta = X_design * beta;
    Eigen::VectorXd mu(eta.size());
    Eigen::VectorXd W(mu.size());
    
    for (int i = 0; i < eta.size(); ++i) {
        mu(i) = std::exp(std::min(20.0, eta(i)));
        W(i) = mu(i);
        if (weights_.size() == y.size()) {
            W(i) *= weights_(i);
        }
    }
    
    Eigen::MatrixXd XtWX = X_design.transpose() * W.asDiagonal() * X_design;
    Eigen::MatrixXd cov_matrix = XtWX.inverse();
    result.standard_errors = cov_matrix.diagonal().array().sqrt();
    
    // Compute p-values and confidence intervals
    result.p_values.resize(p);
    result.confidence_intervals.resize(p);
    
    for (int i = 0; i < p; ++i) {
        double z_stat = beta(i) / result.standard_errors(i);
        double p_val = 2.0 * (1.0 - std::erf(std::abs(z_stat) / std::sqrt(2.0)) / 2.0);
        result.p_values(i) = p_val;
        
        double margin = 1.96 * result.standard_errors(i);
        result.confidence_intervals[i] = {beta(i) - margin, beta(i) + margin};
    }
    
    return result;
}

Eigen::VectorXd PoissonRegression::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd linear_pred = predictLinear(X);
    Eigen::VectorXd predictions(linear_pred.size());
    
    for (int i = 0; i < linear_pred.size(); ++i) {
        predictions(i) = std::exp(std::min(20.0, linear_pred(i)));
    }
    
    return predictions;
}

double PoissonRegression::computeLogLikelihood(const Eigen::MatrixXd& X, 
                                              const Eigen::VectorXd& y) const {
    Eigen::VectorXd mu = predict(X);
    double log_lik = 0.0;
    
    for (int i = 0; i < y.size(); ++i) {
        double mui = std::max(1e-10, mu(i));
        
        // Poisson log-likelihood: y*log(mu) - mu - log(y!)
        log_lik += y(i) * std::log(mui) - mui;
        
        // Subtract log(y!) term
        for (int k = 1; k <= y(i); ++k) {
            log_lik -= std::log(k);
        }
        
        // Apply weights if provided
        if (weights_.size() == y.size()) {
            log_lik *= weights_(i);
        }
    }
    
    return log_lik;
}