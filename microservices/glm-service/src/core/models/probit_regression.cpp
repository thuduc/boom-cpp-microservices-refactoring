#include "probit_regression.hpp"
#include <cmath>
#include <algorithm>

double ProbitRegression::normalCDF(double x) const {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double ProbitRegression::normalPDF(double x) const {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double ProbitRegression::normalQuantile(double p) const {
    // Simple approximation for inverse normal CDF
    // For production, use a more accurate method
    p = std::max(1e-10, std::min(1.0 - 1e-10, p));
    
    // Abramowitz and Stegun approximation
    double a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
    double a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
    double b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
    double b4 = 66.8013118877197, b5 = -13.2806815528857;
    
    double q = p < 0.5 ? p : 1 - p;
    double r = std::sqrt(-std::log(q));
    
    double num = ((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6;
    double den = ((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1;
    
    double x = num / den;
    return p < 0.5 ? -x : x;
}

double ProbitRegression::linkFunction(double p) const {
    // Probit link: Φ^(-1)(p)
    return normalQuantile(p);
}

double ProbitRegression::linkDerivative(double p) const {
    // Derivative of probit: 1 / φ(Φ^(-1)(p))
    double quantile = normalQuantile(p);
    return 1.0 / normalPDF(quantile);
}

double ProbitRegression::inverseLinkFunction(double eta) const {
    // Inverse probit (normal CDF)
    return normalCDF(eta);
}

FitResult ProbitRegression::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    return fitIRLS(X, y);
}

FitResult ProbitRegression::fitIRLS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    FitResult result;
    
    // Check that y contains only 0s and 1s
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) != 0.0 && y(i) != 1.0) {
            throw std::runtime_error("Probit regression requires binary outcomes (0 or 1)");
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
            mu(i) = normalCDF(eta(i));
        }
        
        // Compute weights and working response
        Eigen::VectorXd W(mu.size());
        Eigen::VectorXd z(mu.size());
        
        for (int i = 0; i < mu.size(); ++i) {
            double mui = std::max(1e-10, std::min(1.0 - 1e-10, mu(i)));
            double phi_eta = normalPDF(eta(i));
            
            // For probit: W = φ²(η) / [Φ(η)(1-Φ(η))]
            W(i) = phi_eta * phi_eta / (mui * (1.0 - mui));
            
            // Working response
            z(i) = eta(i) + (y_work(i) - mui) / phi_eta;
            
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
        mu(i) = normalCDF(eta(i));
        double mui = std::max(1e-10, std::min(1.0 - 1e-10, mu(i)));
        double phi_eta = normalPDF(eta(i));
        W(i) = phi_eta * phi_eta / (mui * (1.0 - mui));
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
        double p_val = 2.0 * (1.0 - normalCDF(std::abs(z_stat)));
        result.p_values(i) = p_val;
        
        double margin = 1.96 * result.standard_errors(i);
        result.confidence_intervals[i] = {beta(i) - margin, beta(i) + margin};
    }
    
    return result;
}

Eigen::VectorXd ProbitRegression::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd probs = predictProbability(X);
    Eigen::VectorXd predictions(probs.size());
    
    for (int i = 0; i < probs.size(); ++i) {
        predictions(i) = probs(i) >= 0.5 ? 1.0 : 0.0;
    }
    
    return predictions;
}

Eigen::VectorXd ProbitRegression::predictProbability(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd linear_pred = predictLinear(X);
    Eigen::VectorXd probs(linear_pred.size());
    
    for (int i = 0; i < linear_pred.size(); ++i) {
        probs(i) = normalCDF(linear_pred(i));
    }
    
    return probs;
}

double ProbitRegression::computeLogLikelihood(const Eigen::MatrixXd& X, 
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