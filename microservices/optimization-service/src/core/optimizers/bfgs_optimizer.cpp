#include "../optimization_engine.hpp"
#include <Eigen/Dense>
#include <cmath>

OptimizationResult OptimizationEngine::bfgs(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_point,
    const OptimizationSettings& settings) {
    
    OptimizationResult result;
    result.optimal_point = initial_point;
    result.iterations = 0;
    result.converged = false;
    
    size_t n = initial_point.size();
    std::vector<double> x = initial_point;
    std::vector<double> x_old = x;
    
    // Initialize inverse Hessian approximation to identity
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    
    // Get gradient function
    auto grad_func = settings.use_gradient && settings.gradient_func ? 
        settings.gradient_func : 
        [&objective](const std::vector<double>& pt) { 
            return numericalGradient(objective, pt); 
        };
    
    double f_old = objective(x);
    std::vector<double> g_old = grad_func(x);
    
    result.function_values.push_back(f_old);
    
    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        // Convert gradient to Eigen vector
        Eigen::VectorXd g(n);
        for (size_t i = 0; i < n; ++i) {
            g(i) = g_old[i];
        }
        
        // Search direction
        Eigen::VectorXd p = -H * g;
        
        // Line search
        double alpha = 1.0;
        double c1 = 1e-4;  // Armijo condition parameter
        double rho = 0.5;  // Step size reduction factor
        
        std::vector<double> x_new(n);
        double f_new;
        
        // Backtracking line search
        while (true) {
            for (size_t i = 0; i < n; ++i) {
                x_new[i] = x[i] + alpha * p(i);
            }
            
            f_new = objective(x_new);
            
            if (f_new <= f_old + c1 * alpha * g.dot(p)) {
                break;
            }
            
            alpha *= rho;
            
            if (alpha < 1e-10) {
                result.convergence_reason = "Line search failed";
                result.optimal_value = f_old;
                return result;
            }
        }
        
        // Update position
        x_old = x;
        x = x_new;
        
        // Compute new gradient
        std::vector<double> g_new = grad_func(x);
        
        // Compute differences
        Eigen::VectorXd s(n), y(n), g_new_eigen(n);
        for (size_t i = 0; i < n; ++i) {
            s(i) = x[i] - x_old[i];
            y(i) = g_new[i] - g_old[i];
            g_new_eigen(i) = g_new[i];
        }
        
        // BFGS update
        double sy = s.dot(y);
        if (sy > 1e-10) {
            Eigen::VectorXd Hy = H * y;
            double yHy = y.dot(Hy);
            
            H = H + (sy + yHy) / (sy * sy) * (s * s.transpose()) 
                - 1.0 / sy * (Hy * s.transpose() + s * Hy.transpose());
        }
        
        // Check convergence
        double gradient_norm = g_new_eigen.norm();
        if (gradient_norm < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Gradient norm below tolerance";
            break;
        }
        
        if (std::abs(f_new - f_old) < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Function value change below tolerance";
            break;
        }
        
        // Update for next iteration
        f_old = f_new;
        g_old = g_new;
        result.iterations = iter + 1;
        result.function_values.push_back(f_new);
    }
    
    result.optimal_point = x;
    result.optimal_value = objective(x);
    result.gradient = grad_func(x);
    
    if (!result.converged && result.iterations >= settings.max_iterations) {
        result.convergence_reason = "Maximum iterations reached";
    }
    
    return result;
}