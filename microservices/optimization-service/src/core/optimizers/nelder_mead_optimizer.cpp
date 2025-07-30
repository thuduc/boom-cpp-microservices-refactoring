#include "../optimization_engine.hpp"
#include <algorithm>
#include <numeric>

OptimizationResult OptimizationEngine::nelderMead(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_point,
    const OptimizationSettings& settings) {
    
    OptimizationResult result;
    result.iterations = 0;
    result.converged = false;
    
    size_t n = initial_point.size();
    
    // Parameters for Nelder-Mead
    double alpha = 1.0;    // Reflection
    double gamma = 2.0;    // Expansion
    double rho = 0.5;      // Contraction
    double sigma = 0.5;    // Shrink
    
    // Initialize simplex
    std::vector<std::vector<double>> simplex(n + 1);
    std::vector<double> f_values(n + 1);
    
    simplex[0] = initial_point;
    f_values[0] = objective(initial_point);
    
    // Create initial simplex
    for (size_t i = 0; i < n; ++i) {
        simplex[i + 1] = initial_point;
        simplex[i + 1][i] += settings.simplex_size;
        f_values[i + 1] = objective(simplex[i + 1]);
    }
    
    result.function_values.push_back(f_values[0]);
    
    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        // Sort vertices
        std::vector<size_t> indices(n + 1);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), 
            [&f_values](size_t i, size_t j) { return f_values[i] < f_values[j]; });
        
        // Reorder simplex and function values
        std::vector<std::vector<double>> sorted_simplex(n + 1);
        std::vector<double> sorted_f_values(n + 1);
        for (size_t i = 0; i <= n; ++i) {
            sorted_simplex[i] = simplex[indices[i]];
            sorted_f_values[i] = f_values[indices[i]];
        }
        simplex = sorted_simplex;
        f_values = sorted_f_values;
        
        // Check convergence
        double f_mean = std::accumulate(f_values.begin(), f_values.end(), 0.0) / (n + 1);
        double f_std = 0.0;
        for (double f : f_values) {
            f_std += (f - f_mean) * (f - f_mean);
        }
        f_std = std::sqrt(f_std / (n + 1));
        
        if (f_std < settings.tolerance) {
            result.converged = true;
            result.convergence_reason = "Function values converged";
            break;
        }
        
        // Calculate centroid (excluding worst point)
        std::vector<double> centroid(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                centroid[j] += simplex[i][j];
            }
        }
        for (size_t j = 0; j < n; ++j) {
            centroid[j] /= n;
        }
        
        // Reflection
        std::vector<double> x_r(n);
        for (size_t j = 0; j < n; ++j) {
            x_r[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j]);
        }
        double f_r = objective(x_r);
        
        if (f_values[0] <= f_r && f_r < f_values[n - 1]) {
            // Accept reflection
            simplex[n] = x_r;
            f_values[n] = f_r;
        } else if (f_r < f_values[0]) {
            // Try expansion
            std::vector<double> x_e(n);
            for (size_t j = 0; j < n; ++j) {
                x_e[j] = centroid[j] + gamma * (x_r[j] - centroid[j]);
            }
            double f_e = objective(x_e);
            
            if (f_e < f_r) {
                simplex[n] = x_e;
                f_values[n] = f_e;
            } else {
                simplex[n] = x_r;
                f_values[n] = f_r;
            }
        } else {
            // Contraction
            if (f_r < f_values[n]) {
                // Outside contraction
                std::vector<double> x_c(n);
                for (size_t j = 0; j < n; ++j) {
                    x_c[j] = centroid[j] + rho * (x_r[j] - centroid[j]);
                }
                double f_c = objective(x_c);
                
                if (f_c <= f_r) {
                    simplex[n] = x_c;
                    f_values[n] = f_c;
                } else {
                    // Shrink
                    for (size_t i = 1; i <= n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        f_values[i] = objective(simplex[i]);
                    }
                }
            } else {
                // Inside contraction
                std::vector<double> x_c(n);
                for (size_t j = 0; j < n; ++j) {
                    x_c[j] = centroid[j] + rho * (simplex[n][j] - centroid[j]);
                }
                double f_c = objective(x_c);
                
                if (f_c < f_values[n]) {
                    simplex[n] = x_c;
                    f_values[n] = f_c;
                } else {
                    // Shrink
                    for (size_t i = 1; i <= n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        }
                        f_values[i] = objective(simplex[i]);
                    }
                }
            }
        }
        
        result.iterations = iter + 1;
        result.function_values.push_back(*std::min_element(f_values.begin(), f_values.end()));
    }
    
    // Find best vertex
    auto min_it = std::min_element(f_values.begin(), f_values.end());
    size_t best_idx = std::distance(f_values.begin(), min_it);
    
    result.optimal_point = simplex[best_idx];
    result.optimal_value = f_values[best_idx];
    
    if (!result.converged && result.iterations >= settings.max_iterations) {
        result.convergence_reason = "Maximum iterations reached";
    }
    
    return result;
}