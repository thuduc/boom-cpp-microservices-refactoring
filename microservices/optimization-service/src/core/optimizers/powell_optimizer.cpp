#include "../optimization_engine.hpp"
#include <cmath>
#include <limits>

// Line minimization helper
static double lineMinimize(
    const ObjectiveFunction& objective,
    const std::vector<double>& point,
    const std::vector<double>& direction,
    double initial_step = 1.0) {
    
    // Golden section search
    const double golden = (3.0 - std::sqrt(5.0)) / 2.0;
    const double tol = 1e-8;
    
    // Bracket the minimum
    double a = 0.0;
    double b = initial_step;
    
    auto f = [&](double t) {
        std::vector<double> x(point.size());
        for (size_t i = 0; i < point.size(); ++i) {
            x[i] = point[i] + t * direction[i];
        }
        return objective(x);
    };
    
    double fa = f(a);
    double fb = f(b);
    
    // Expand bracket if needed
    if (fb > fa) {
        double c = b;
        b = a;
        a = c;
        double tmp = fa;
        fa = fb;
        fb = tmp;
    }
    
    // Golden section search
    double c = a + golden * (b - a);
    double d = a + (1.0 - golden) * (b - a);
    
    while (std::abs(b - a) > tol) {
        double fc = f(c);
        double fd = f(d);
        
        if (fc < fd) {
            b = d;
            d = c;
            c = a + golden * (b - a);
        } else {
            a = c;
            c = d;
            d = a + (1.0 - golden) * (b - a);
        }
    }
    
    return (a + b) / 2.0;
}

OptimizationResult OptimizationEngine::powell(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_point,
    const OptimizationSettings& settings) {
    
    OptimizationResult result;
    result.iterations = 0;
    result.converged = false;
    
    size_t n = initial_point.size();
    std::vector<double> x = initial_point;
    
    // Initialize search directions to coordinate directions
    std::vector<std::vector<double>> directions(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        directions[i][i] = 1.0;
    }
    
    double f_old = objective(x);
    result.function_values.push_back(f_old);
    
    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        std::vector<double> x_start = x;
        double f_start = f_old;
        
        // Minimize along each direction
        std::vector<double> delta_f(n);
        int biggest_decrease_idx = 0;
        double biggest_decrease = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            double f_before = objective(x);
            
            // Line minimization along direction i
            double alpha = lineMinimize(objective, x, directions[i]);
            
            // Update position
            for (size_t j = 0; j < n; ++j) {
                x[j] += alpha * directions[i][j];
            }
            
            double f_after = objective(x);
            delta_f[i] = f_before - f_after;
            
            if (delta_f[i] > biggest_decrease) {
                biggest_decrease = delta_f[i];
                biggest_decrease_idx = i;
            }
        }
        
        // Compute new direction
        std::vector<double> new_direction(n);
        for (size_t i = 0; i < n; ++i) {
            new_direction[i] = x[i] - x_start[i];
        }
        
        // Check convergence
        double step_size = 0.0;
        for (size_t i = 0; i < n; ++i) {
            step_size += new_direction[i] * new_direction[i];
        }
        step_size = std::sqrt(step_size);
        
        if (step_size < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Step size below tolerance";
            break;
        }
        
        double f_new = objective(x);
        if (std::abs(f_new - f_start) < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Function value change below tolerance";
            break;
        }
        
        // Powell's modification to maintain linear independence
        if (biggest_decrease > 0) {
            // Extrapolate
            std::vector<double> x_extrap(n);
            for (size_t i = 0; i < n; ++i) {
                x_extrap[i] = 2.0 * x[i] - x_start[i];
            }
            double f_extrap = objective(x_extrap);
            
            if (f_extrap < f_start) {
                double t = 2.0 * (f_start - 2.0 * f_new + f_extrap);
                double temp = f_start - f_new - biggest_decrease;
                t = t * temp * temp;
                temp = f_start - f_extrap;
                double s = biggest_decrease * temp * temp;
                
                if (t < s) {
                    // Replace direction with new conjugate direction
                    double alpha = lineMinimize(objective, x, new_direction);
                    for (size_t i = 0; i < n; ++i) {
                        x[i] += alpha * new_direction[i];
                    }
                    
                    // Update directions
                    directions[biggest_decrease_idx] = new_direction;
                }
            }
        }
        
        f_old = objective(x);
        result.iterations = iter + 1;
        result.function_values.push_back(f_old);
    }
    
    result.optimal_point = x;
    result.optimal_value = objective(x);
    
    if (!result.converged && result.iterations >= settings.max_iterations) {
        result.convergence_reason = "Maximum iterations reached";
    }
    
    return result;
}