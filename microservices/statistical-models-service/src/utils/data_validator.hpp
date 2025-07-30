#ifndef DATA_VALIDATOR_HPP
#define DATA_VALIDATOR_HPP

#include <vector>
#include <string>

class DataValidator {
public:
    // Validate data for specific model types
    static bool validateData(const std::vector<double>& data, const std::string& model_type);
    
private:
    static bool validateGaussianData(const std::vector<double>& data);
    static bool validateGammaData(const std::vector<double>& data);
    static bool validateBetaData(const std::vector<double>& data);
    static bool validatePoissonData(const std::vector<double>& data);
    static bool validateMultinomialData(const std::vector<double>& data);
};

#endif // DATA_VALIDATOR_HPP