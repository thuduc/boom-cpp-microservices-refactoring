#ifndef REGRESSION_DIAGNOSTICS_HPP
#define REGRESSION_DIAGNOSTICS_HPP

#include "../glm_base.hpp"
#include <Eigen/Dense>

struct DiagnosticsResult {
    // Residuals
    Eigen::VectorXd residuals;
    Eigen::VectorXd standardized_residuals;
    Eigen::VectorXd deviance_residuals;
    
    // Goodness of fit
    double r_squared;
    double adjusted_r_squared;
    double deviance;
    double pearson_chi_squared;
    
    // Influence measures
    Eigen::VectorXd leverage;
    Eigen::VectorXd cooks_distance;
    
    // Tests
    double durbin_watson;
    double breusch_pagan_p_value;
    double shapiro_wilk_p_value;
};

class RegressionDiagnostics {
public:
    // Compute all diagnostics
    DiagnosticsResult compute(const GLMBase* model, 
                             const Eigen::MatrixXd& X, 
                             const Eigen::VectorXd& y);
    
private:
    // Residual computations
    Eigen::VectorXd computeRawResiduals(const GLMBase* model,
                                       const Eigen::MatrixXd& X,
                                       const Eigen::VectorXd& y);
    
    Eigen::VectorXd computeStandardizedResiduals(const Eigen::VectorXd& residuals,
                                                 const Eigen::MatrixXd& hat_matrix);
    
    Eigen::VectorXd computeDevianceResiduals(const GLMBase* model,
                                           const Eigen::MatrixXd& X,
                                           const Eigen::VectorXd& y);
    
    // Goodness of fit measures
    double computeRSquared(const Eigen::VectorXd& y,
                          const Eigen::VectorXd& fitted);
    
    double computeAdjustedRSquared(double r_squared, int n, int p);
    
    double computeDeviance(const GLMBase* model,
                          const Eigen::MatrixXd& X,
                          const Eigen::VectorXd& y);
    
    // Influence measures
    Eigen::MatrixXd computeHatMatrix(const Eigen::MatrixXd& X,
                                    const Eigen::VectorXd& weights);
    
    Eigen::VectorXd computeCooksDistance(const Eigen::VectorXd& std_residuals,
                                       const Eigen::VectorXd& leverage,
                                       int p);
    
    // Statistical tests
    double computeDurbinWatson(const Eigen::VectorXd& residuals);
    
    double computeBreuschPagan(const Eigen::VectorXd& residuals,
                              const Eigen::MatrixXd& X);
    
    double computeShapiroWilk(const Eigen::VectorXd& residuals);
    
    // Helper functions
    double chiSquaredCDF(double x, int df);
    double normalCDF(double x);
};

#endif // REGRESSION_DIAGNOSTICS_HPP