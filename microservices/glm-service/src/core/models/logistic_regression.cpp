#include "logistic_regression.hpp"
#include <cmath>
#include <algorithm>

double LogisticRegression::sigmoid(double x) const {
    // Numerically stable sigmoid
    if (x >= 0) {
        double exp_neg_x = std::exp(-x);
        return 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

double LogisticRegression::linkFunction(double p) const {
    // Logit link: log(p / (1 - p))
    p = std::max(1e-10, std::min(1.0 - 1e-10, p));
    return std::log(p / (1.0 - p));
}

double LogisticRegression::linkDerivative(double p) const {
    // Derivative of logit: 1 / (p * (1 - p))
    p = std::max(1e-10, std::min(1.0 - 1e-10, p));
    return 1.0 / (p * (1.0 - p));
}

double LogisticRegression::inverseLinkFunction(double eta) const {
    // Inverse logit (sigmoid)
    return sigmoid(eta);
}

FitResult LogisticRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    return fitIRLS(X, y);
}

FitResult LogisticRegression::fitIRLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    FitResult result;
    
    // Check that y contains only 0s and 1s
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) != 0.0 && y(i) != 1.0) {
            throw std::runtime_error("Logistic regression requires binary outcomes (0 or 1)");
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
    
    // IRLS algorithm
    const int max_iter = 100;
    const double tol = 1e-6;
    bool converged = false;
    int iter = 0;
    
    for (iter = 0; iter < max_iter; ++iter) {
        // Compute linear predictor
        Eigen::VectorXd eta = X_design * beta;
        
        // Compute probabilities
        Eigen::VectorXd mu(eta.size());
        for (int i = 0; i < eta.size(); ++i) {
            mu(i) = sigmoid(eta(i));
        }
        
        // Compute weights and working response
        Eigen::VectorXd W(mu.size());
        Eigen::VectorXd z(mu.size());
        
        for (int i = 0; i < mu.size(); ++i) {
            double mui = std::max(1e-10, std::min(1.0 - 1e-10, mu(i)));
            W(i) = mui * (1.0 - mui);
            z(i) = eta(i) + (y_work(i) - mui) / W(i);
            
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
    
    // Compute standard errors (from final weighted covariance matrix)
    Eigen::VectorXd eta = X_design * beta;
    Eigen::VectorXd mu(eta.size());
    Eigen::VectorXd W(mu.size());
    
    for (int i = 0; i < eta.size(); ++i) {
        mu(i) = sigmoid(eta(i));
        W(i) = mu(i) * (1.0 - mu(i));
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

Eigen::VectorXd LogisticRegression::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd probs = predictProbability(X);
    Eigen::VectorXd predictions(probs.size());
    
    for (int i = 0; i < probs.size(); ++i) {
        predictions(i) = probs(i) >= 0.5 ? 1.0 : 0.0;
    }
    
    return predictions;
}

Eigen::VectorXd LogisticRegression::predictProbability(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd linear_pred = predictLinear(X);
    Eigen::VectorXd probs(linear_pred.size());
    
    for (int i = 0; i < linear_pred.size(); ++i) {
        probs(i) = sigmoid(linear_pred(i));
    }
    
    return probs;
}

double LogisticRegression::computeLogLikelihood(const Eigen::MatrixXd& X, 
                                               const Eigen::VectorXd& y) const {
    Eigen::VectorXd probs = predictProbability(X);
    double log_lik = 0.0;
    
    for (int i = 0; i < y.size(); ++i) {
        double pi = std::max(1e-10, std::min(1.0 - 1e-10, probs(i)));
        if (y(i) == 1.0) {
            log_lik += std::log(pi);
        } else {
            log_lik += std::log(1.0 - pi);
        }
        
        // Apply weights if provided
        if (weights_.size() == y.size()) {
            log_lik *= weights_(i);
        }
    }
    
    return log_lik;
}