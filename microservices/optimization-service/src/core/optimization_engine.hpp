#ifndef OPTIMIZATION_ENGINE_HPP
#define OPTIMIZATION_ENGINE_HPP

#include <vector>
#include <functional>
#include <Eigen/Dense>

// Type definitions for optimization functions
using ObjectiveFunction = std::function<double(const std::vector<double>&)>;
using GradientFunction = std::function<std::vector<double>(const std::vector<double>&)>;
using HessianFunction = std::function<Eigen::MatrixXd(const std::vector<double>&)>;

struct OptimizationSettings {
    int max_iterations = 1000;
    double tolerance = 1e-6;
    bool use_gradient = false;
    
    // Optional functions
    GradientFunction gradient_func;
    HessianFunction hessian_func;
    
    // Method-specific parameters
    double simplex_size = 0.1;           // Nelder-Mead
    double initial_temperature = 1.0;     // Simulated Annealing
    double cooling_rate = 0.95;          // Simulated Annealing
    double step_size = 0.1;              // Various methods
    
    // Bounds (optional)
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
};

struct OptimizationResult {
    std::vector<double> optimal_point;
    double optimal_value;
    int iterations;
    bool converged;
    std::string convergence_reason;
    std::vector<double> gradient;  // Final gradient if available
    std::vector<double> function_values;  // History of function values
};

class OptimizationEngine {
public:
    // BFGS quasi-Newton method
    static OptimizationResult bfgs(
        const ObjectiveFunction& objective,
        const std::vector<double>& initial_point,
        const OptimizationSettings& settings);
    
    // Nelder-Mead simplex method
    static OptimizationResult nelderMead(
        const ObjectiveFunction& objective,
        const std::vector<double>& initial_point,
        const OptimizationSettings& settings);
    
    // Powell's conjugate direction method
    static OptimizationResult powell(
        const ObjectiveFunction& objective,
        const std::vector<double>& initial_point,
        const OptimizationSettings& settings);
    
    // Simulated annealing
    static OptimizationResult simulatedAnnealing(
        const ObjectiveFunction& objective,
        const std::vector<double>& initial_point,
        const OptimizationSettings& settings);
    
    // Newton's method
    static OptimizationResult newton(
        const ObjectiveFunction& objective,
        const std::vector<double>& initial_point,
        const OptimizationSettings& settings);
    
private:
    // Helper functions
    static std::vector<double> numericalGradient(
        const ObjectiveFunction& objective,
        const std::vector<double>& point,
        double epsilon = 1e-8);
    
    static Eigen::MatrixXd numericalHessian(
        const ObjectiveFunction& objective,
        const std::vector<double>& point,
        double epsilon = 1e-5);
};

#endif // OPTIMIZATION_ENGINE_HPP