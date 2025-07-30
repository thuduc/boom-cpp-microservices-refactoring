#include "../optimization_engine.hpp"
#include <random>
#include <cmath>

OptimizationResult OptimizationEngine::simulatedAnnealing(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_point,
    const OptimizationSettings& settings) {
    
    OptimizationResult result;
    result.iterations = 0;
    result.converged = false;
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::normal_distribution<> normal(0.0, 1.0);
    
    size_t n = initial_point.size();
    std::vector<double> current = initial_point;
    std::vector<double> best = current;
    
    double current_value = objective(current);
    double best_value = current_value;
    
    double temperature = settings.initial_temperature;
    
    result.function_values.push_back(current_value);
    
    // Check if bounds are provided
    bool has_bounds = !settings.lower_bounds.empty() && !settings.upper_bounds.empty();
    
    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        // Generate neighbor
        std::vector<double> neighbor = current;
        
        for (size_t i = 0; i < n; ++i) {
            neighbor[i] += settings.step_size * normal(gen);
            
            // Apply bounds if provided
            if (has_bounds) {
                neighbor[i] = std::max(settings.lower_bounds[i], 
                                     std::min(settings.upper_bounds[i], neighbor[i]));
            }
        }
        
        double neighbor_value = objective(neighbor);
        double delta = neighbor_value - current_value;
        
        // Accept or reject the neighbor
        if (delta < 0 || uniform(gen) < std::exp(-delta / temperature)) {
            current = neighbor;
            current_value = neighbor_value;
            
            // Update best if needed
            if (current_value < best_value) {
                best = current;
                best_value = current_value;
            }
        }
        
        // Cool down
        temperature *= settings.cooling_rate;
        
        // Check convergence
        if (temperature < 1e-10) {
            result.converged = true;
            result.convergence_reason = "Temperature below threshold";
            break;
        }
        
        result.iterations = iter + 1;
        result.function_values.push_back(best_value);
    }
    
    result.optimal_point = best;
    result.optimal_value = best_value;
    
    if (!result.converged && result.iterations >= settings.max_iterations) {
        result.convergence_reason = "Maximum iterations reached";
    }
    
    return result;
}