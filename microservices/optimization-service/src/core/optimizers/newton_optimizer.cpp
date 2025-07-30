#include "../optimization_engine.hpp"
#include <Eigen/Dense>
#include <cmath>

OptimizationResult OptimizationEngine::newton(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_point,
    const OptimizationSettings& settings) {
    
    OptimizationResult result;
    result.iterations = 0;
    result.converged = false;
    
    size_t n = initial_point.size();
    std::vector<double> x = initial_point;
    
    double f_old = objective(x);
    result.function_values.push_back(f_old);
    
    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        // Compute gradient
        std::vector<double> grad = settings.gradient_func(x);
        
        // Compute Hessian
        Eigen::MatrixXd H = settings.hessian_func(x);
        
        // Convert gradient to Eigen vector
        Eigen::VectorXd g(n);
        for (size_t i = 0; i < n; ++i) {
            g(i) = grad[i];
        }
        
        // Check gradient norm for convergence
        double grad_norm = g.norm();
        if (grad_norm < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Gradient norm below tolerance";
            break;
        }
        
        // Solve H * p = -g for Newton direction p
        Eigen::VectorXd p;
        try {
            // Use LLT decomposition if Hessian is positive definite
            Eigen::LLT<Eigen::MatrixXd> llt(H);
            if (llt.info() == Eigen::Success) {
                p = llt.solve(-g);
            } else {
                // Fall back to LU decomposition
                p = H.lu().solve(-g);
            }
        } catch (const std::exception& e) {
            result.convergence_reason = "Failed to solve Newton system";
            break;
        }
        
        // Line search with backtracking
        double alpha = 1.0;
        double c1 = 1e-4;
        double rho = 0.5;
        
        std::vector<double> x_new(n);
        double f_new;
        
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
                result.optimal_point = x;
                result.optimal_value = f_old;
                return result;
            }
        }
        
        // Check function value change
        if (std::abs(f_new - f_old) < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Function value change below tolerance";
            x = x_new;
            break;
        }
        
        // Update
        x = x_new;
        f_old = f_new;
        result.iterations = iter + 1;
        result.function_values.push_back(f_new);
    }
    
    result.optimal_point = x;
    result.optimal_value = objective(x);
    result.gradient = settings.gradient_func(x);
    
    if (!result.converged && result.iterations >= settings.max_iterations) {
        result.convergence_reason = "Maximum iterations reached";
    }
    
    return result;
}