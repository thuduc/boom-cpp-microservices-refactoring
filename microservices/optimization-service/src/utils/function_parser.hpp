#ifndef FUNCTION_PARSER_HPP
#define FUNCTION_PARSER_HPP

#include <string>
#include <functional>
#include <vector>
#include <nlohmann/json.hpp>
#include "../core/optimization_engine.hpp"

using json = nlohmann::json;

class FunctionParser {
public:
    // Parse objective function from JSON
    static ObjectiveFunction parseObjective(const json& func_json);
    
    // Parse gradient function from JSON
    static GradientFunction parseGradient(const json& grad_json);
    
    // Parse Hessian function from JSON
    static HessianFunction parseHessian(const json& hess_json);
    
private:
    // Helper to evaluate simple mathematical expressions
    static double evaluateExpression(const std::string& expr, 
                                   const std::vector<double>& variables);
};

#endif // FUNCTION_PARSER_HPP