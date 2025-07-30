#include "data_validator.hpp"
#include <algorithm>
#include <cmath>

bool DataValidator::validateData(const std::vector<double>& data, const std::string& model_type) {
    if (data.empty()) return false;
    
    if (model_type == "gaussian" || model_type == "normal") {
        return validateGaussianData(data);
    } else if (model_type == "gamma") {
        return validateGammaData(data);
    } else if (model_type == "beta") {
        return validateBetaData(data);
    } else if (model_type == "poisson") {
        return validatePoissonData(data);
    } else if (model_type == "multinomial") {
        return validateMultinomialData(data);
    }
    
    return false;
}

bool DataValidator::validateGaussianData(const std::vector<double>& data) {
    // Gaussian data can be any real numbers
    // Check for NaN or infinity
    return std::all_of(data.begin(), data.end(), 
        [](double x) { return std::isfinite(x); });
}

bool DataValidator::validateGammaData(const std::vector<double>& data) {
    // Gamma data must be positive
    return std::all_of(data.begin(), data.end(), 
        [](double x) { return std::isfinite(x) && x > 0; });
}

bool DataValidator::validateBetaData(const std::vector<double>& data) {
    // Beta data must be in (0, 1)
    return std::all_of(data.begin(), data.end(), 
        [](double x) { return std::isfinite(x) && x > 0 && x < 1; });
}

bool DataValidator::validatePoissonData(const std::vector<double>& data) {
    // Poisson data must be non-negative integers
    return std::all_of(data.begin(), data.end(), 
        [](double x) { 
            return std::isfinite(x) && x >= 0 && 
                   std::floor(x) == x;  // Check if integer
        });
}

bool DataValidator::validateMultinomialData(const std::vector<double>& data) {
    // Multinomial data must be non-negative integers (category indices)
    return std::all_of(data.begin(), data.end(), 
        [](double x) { 
            return std::isfinite(x) && x >= 0 && 
                   std::floor(x) == x;  // Check if integer
        });
}