#include "regression_diagnostics.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

DiagnosticsResult RegressionDiagnostics::compute(const GLMBase* model, 
                                                 const Eigen::MatrixXd& X, 
                                                 const Eigen::VectorXd& y) {
    DiagnosticsResult result;
    
    // Compute fitted values
    Eigen::VectorXd fitted = model->predict(X);
    
    // Compute residuals
    result.residuals = computeRawResiduals(model, X, y);
    
    // For GLMs, we need weights for the hat matrix
    // For now, use uniform weights
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(y.size());
    
    // Compute hat matrix and leverage
    Eigen::MatrixXd hat_matrix = computeHatMatrix(X, weights);
    result.leverage = hat_matrix.diagonal();
    
    // Standardized residuals
    result.standardized_residuals = computeStandardizedResiduals(result.residuals, hat_matrix);
    
    // Deviance residuals
    result.deviance_residuals = computeDevianceResiduals(model, X, y);
    
    // Goodness of fit
    result.r_squared = computeRSquared(y, fitted);
    int p = X.cols() + (model->getIntercept() != 0 ? 1 : 0);
    result.adjusted_r_squared = computeAdjustedRSquared(result.r_squared, y.size(), p);
    result.deviance = computeDeviance(model, X, y);
    
    // Pearson chi-squared (sum of squared standardized residuals)
    result.pearson_chi_squared = result.standardized_residuals.squaredNorm();
    
    // Influence measures
    result.cooks_distance = computeCooksDistance(result.standardized_residuals, 
                                                 result.leverage, p);
    
    // Statistical tests
    result.durbin_watson = computeDurbinWatson(result.residuals);
    result.breusch_pagan_p_value = computeBreuschPagan(result.residuals, X);
    result.shapiro_wilk_p_value = computeShapiroWilk(result.residuals);
    
    return result;
}

Eigen::VectorXd RegressionDiagnostics::computeRawResiduals(const GLMBase* model,
                                                          const Eigen::MatrixXd& X,
                                                          const Eigen::VectorXd& y) {
    Eigen::VectorXd fitted = model->predict(X);
    return y - fitted;
}

Eigen::VectorXd RegressionDiagnostics::computeStandardizedResiduals(
    const Eigen::VectorXd& residuals,
    const Eigen::MatrixXd& hat_matrix) {
    
    Eigen::VectorXd std_residuals(residuals.size());
    double sigma = std::sqrt(residuals.squaredNorm() / (residuals.size() - hat_matrix.trace()));
    
    for (int i = 0; i < residuals.size(); ++i) {
        double var_factor = 1.0 - hat_matrix(i, i);
        if (var_factor > 1e-10) {
            std_residuals(i) = residuals(i) / (sigma * std::sqrt(var_factor));
        } else {
            std_residuals(i) = 0.0;
        }
    }
    
    return std_residuals;
}

Eigen::VectorXd RegressionDiagnostics::computeDevianceResiduals(
    const GLMBase* model,
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y) {
    
    // For linear regression, deviance residuals = standardized residuals
    // For other GLMs, this would be model-specific
    return computeRawResiduals(model, X, y);
}

double RegressionDiagnostics::computeRSquared(const Eigen::VectorXd& y,
                                             const Eigen::VectorXd& fitted) {
    double y_mean = y.mean();
    double ss_tot = (y.array() - y_mean).square().sum();
    double ss_res = (y - fitted).squaredNorm();
    
    if (ss_tot < 1e-10) return 0.0;
    return 1.0 - ss_res / ss_tot;
}

double RegressionDiagnostics::computeAdjustedRSquared(double r_squared, int n, int p) {
    if (n <= p) return 0.0;
    return 1.0 - (1.0 - r_squared) * (n - 1) / (n - p);
}

double RegressionDiagnostics::computeDeviance(const GLMBase* model,
                                            const Eigen::MatrixXd& X,
                                            const Eigen::VectorXd& y) {
    // For now, return residual sum of squares
    // Model-specific implementations would override this
    Eigen::VectorXd residuals = computeRawResiduals(model, X, y);
    return residuals.squaredNorm();
}

Eigen::MatrixXd RegressionDiagnostics::computeHatMatrix(const Eigen::MatrixXd& X,
                                                       const Eigen::VectorXd& weights) {
    // Add intercept column
    Eigen::MatrixXd X_design(X.rows(), X.cols() + 1);
    X_design.col(0) = Eigen::VectorXd::Ones(X.rows());
    X_design.rightCols(X.cols()) = X;
    
    // Weight matrix
    Eigen::MatrixXd W = weights.asDiagonal();
    
    // Hat matrix: X(X'WX)^(-1)X'W
    Eigen::MatrixXd XtWX = X_design.transpose() * W * X_design;
    Eigen::MatrixXd XtWX_inv = XtWX.inverse();
    
    return X_design * XtWX_inv * X_design.transpose() * W;
}

Eigen::VectorXd RegressionDiagnostics::computeCooksDistance(
    const Eigen::VectorXd& std_residuals,
    const Eigen::VectorXd& leverage,
    int p) {
    
    Eigen::VectorXd cooks_d(std_residuals.size());
    
    for (int i = 0; i < std_residuals.size(); ++i) {
        if (1.0 - leverage(i) > 1e-10) {
            cooks_d(i) = std_residuals(i) * std_residuals(i) * leverage(i) / 
                        (p * (1.0 - leverage(i)));
        } else {
            cooks_d(i) = 0.0;
        }
    }
    
    return cooks_d;
}

double RegressionDiagnostics::computeDurbinWatson(const Eigen::VectorXd& residuals) {
    if (residuals.size() < 2) return 2.0;
    
    double numerator = 0.0;
    for (int i = 1; i < residuals.size(); ++i) {
        double diff = residuals(i) - residuals(i-1);
        numerator += diff * diff;
    }
    
    double denominator = residuals.squaredNorm();
    
    if (denominator < 1e-10) return 2.0;
    return numerator / denominator;
}

double RegressionDiagnostics::computeBreuschPagan(const Eigen::VectorXd& residuals,
                                                 const Eigen::MatrixXd& X) {
    // Simplified Breusch-Pagan test
    // Regress squared residuals on X
    Eigen::VectorXd squared_res = residuals.array().square();
    
    // Add intercept
    Eigen::MatrixXd X_design(X.rows(), X.cols() + 1);
    X_design.col(0) = Eigen::VectorXd::Ones(X.rows());
    X_design.rightCols(X.cols()) = X;
    
    // OLS regression
    Eigen::VectorXd coeffs = (X_design.transpose() * X_design).ldlt().solve(
        X_design.transpose() * squared_res);
    
    // Compute R-squared
    Eigen::VectorXd fitted = X_design * coeffs;
    double mean_sq_res = squared_res.mean();
    double ss_tot = (squared_res.array() - mean_sq_res).square().sum();
    double ss_res = (squared_res - fitted).squaredNorm();
    double r_squared = 1.0 - ss_res / ss_tot;
    
    // Test statistic: n * R²
    double test_stat = residuals.size() * r_squared;
    
    // Chi-squared p-value with df = p - 1
    int df = X.cols();
    return 1.0 - chiSquaredCDF(test_stat, df);
}

double RegressionDiagnostics::computeShapiroWilk(const Eigen::VectorXd& residuals) {
    // Simplified normality test
    // For a full implementation, use the Shapiro-Wilk algorithm
    // Here we use a simplified approach based on skewness and kurtosis
    
    int n = residuals.size();
    if (n < 3) return 1.0;
    
    double mean = residuals.mean();
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double d = residuals(i) - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    double skewness = m3 / std::pow(m2, 1.5);
    double kurtosis = m4 / (m2 * m2) - 3.0;
    
    // Jarque-Bera statistic as approximation
    double jb_stat = n * (skewness * skewness / 6.0 + kurtosis * kurtosis / 24.0);
    
    // Chi-squared p-value with df = 2
    return 1.0 - chiSquaredCDF(jb_stat, 2);
}

double RegressionDiagnostics::chiSquaredCDF(double x, int df) {
    // Simplified chi-squared CDF using gamma function
    // For production, use a proper implementation
    if (x <= 0) return 0.0;
    
    // Use normal approximation for large df
    if (df > 30) {
        double z = (std::pow(x / df, 1.0/3.0) - (1.0 - 2.0/(9.0*df))) / 
                   std::sqrt(2.0/(9.0*df));
        return normalCDF(z);
    }
    
    // Simple approximation for small df
    double p = std::exp(-x/2.0);
    double term = p;
    double sum = term;
    
    for (int k = 1; k < df/2; ++k) {
        term *= x / (2.0 * k);
        sum += term;
    }
    
    return 1.0 - sum;
}

double RegressionDiagnostics::normalCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}