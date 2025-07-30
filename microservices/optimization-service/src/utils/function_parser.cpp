#include "function_parser.hpp"
#include <cmath>
#include <stdexcept>

ObjectiveFunction FunctionParser::parseObjective(const json& func_json) {
    if (func_json.is_string()) {
        // Simple expression format
        std::string expr = func_json.get<std::string>();
        
        // Common test functions
        if (expr == "rosenbrock") {
            return [](const std::vector<double>& x) {
                double sum = 0.0;
                for (size_t i = 0; i < x.size() - 1; ++i) {
                    double t1 = x[i + 1] - x[i] * x[i];
                    double t2 = 1.0 - x[i];
                    sum += 100.0 * t1 * t1 + t2 * t2;
                }
                return sum;
            };
        } else if (expr == "sphere") {
            return [](const std::vector<double>& x) {
                double sum = 0.0;
                for (double xi : x) {
                    sum += xi * xi;
                }
                return sum;
            };
        } else if (expr == "rastrigin") {
            return [](const std::vector<double>& x) {
                double sum = 10.0 * x.size();
                for (double xi : x) {
                    sum += xi * xi - 10.0 * std::cos(2.0 * M_PI * xi);
                }
                return sum;
            };
        }
    } else if (func_json.is_object()) {
        // Polynomial format: {"coefficients": [...], "powers": [[...]]}
        if (func_json.contains("coefficients") && func_json.contains("powers")) {
            std::vector<double> coeffs = func_json["coefficients"];
            std::vector<std::vector<int>> powers = func_json["powers"];
            
            return [coeffs, powers](const std::vector<double>& x) {
                double sum = 0.0;
                for (size_t i = 0; i < coeffs.size(); ++i) {
                    double term = coeffs[i];
                    for (size_t j = 0; j < x.size() && j < powers[i].size(); ++j) {
                        term *= std::pow(x[j], powers[i][j]);
                    }
                    sum += term;
                }
                return sum;
            };
        }
    }
    
    throw std::invalid_argument("Invalid objective function format");
}

GradientFunction FunctionParser::parseGradient(const json& grad_json) {
    if (grad_json.is_string()) {
        std::string expr = grad_json.get<std::string>();
        
        if (expr == "rosenbrock") {
            return [](const std::vector<double>& x) {
                std::vector<double> grad(x.size(), 0.0);
                
                for (size_t i = 0; i < x.size(); ++i) {
                    if (i < x.size() - 1) {
                        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
                    }
                    if (i > 0) {
                        grad[i] += 200.0 * (x[i] - x[i - 1] * x[i - 1]);
                    }
                }
                
                return grad;
            };
        } else if (expr == "sphere") {
            return [](const std::vector<double>& x) {
                std::vector<double> grad(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    grad[i] = 2.0 * x[i];
                }
                return grad;
            };
        }
    }
    
    throw std::invalid_argument("Invalid gradient function format");
}

HessianFunction FunctionParser::parseHessian(const json& hess_json) {
    if (hess_json.is_string()) {
        std::string expr = hess_json.get<std::string>();
        
        if (expr == "sphere") {
            return [](const std::vector<double>& x) {
                size_t n = x.size();
                Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
                for (size_t i = 0; i < n; ++i) {
                    H(i, i) = 2.0;
                }
                return H;
            };
        }
    }
    
    throw std::invalid_argument("Invalid Hessian function format");
}