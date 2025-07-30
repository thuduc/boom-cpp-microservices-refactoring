#include "core/optimization_engine.hpp"
#include <cmath>
#include <algorithm>

std::vector<double> OptimizationEngine::numericalGradient(
    const ObjectiveFunction& objective,
    const std::vector<double>& point,
    double epsilon) {
    
    std::vector<double> gradient(point.size());
    std::vector<double> point_plus = point;
    std::vector<double> point_minus = point;
    
    for (size_t i = 0; i < point.size(); ++i) {
        point_plus[i] += epsilon;
        point_minus[i] -= epsilon;
        
        gradient[i] = (objective(point_plus) - objective(point_minus)) / (2 * epsilon);
        
        point_plus[i] = point[i];
        point_minus[i] = point[i];
    }
    
    return gradient;
}

Eigen::MatrixXd OptimizationEngine::numericalHessian(
    const ObjectiveFunction& objective,
    const std::vector<double>& point,
    double epsilon) {
    
    size_t n = point.size();
    Eigen::MatrixXd hessian(n, n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::vector<double> point_pp = point;
            std::vector<double> point_pm = point;
            std::vector<double> point_mp = point;
            std::vector<double> point_mm = point;
            
            point_pp[i] += epsilon;
            point_pp[j] += epsilon;
            
            point_pm[i] += epsilon;
            point_pm[j] -= epsilon;
            
            point_mp[i] -= epsilon;
            point_mp[j] += epsilon;
            
            point_mm[i] -= epsilon;
            point_mm[j] -= epsilon;
            
            hessian(i, j) = (objective(point_pp) - objective(point_pm) - 
                            objective(point_mp) + objective(point_mm)) / 
                           (4 * epsilon * epsilon);
        }
    }
    
    return hessian;
}