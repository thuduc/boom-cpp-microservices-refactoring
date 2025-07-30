#ifndef DATA_TRANSFORMER_HPP
#define DATA_TRANSFORMER_HPP

#include <vector>
#include <string>

class DataTransformer {
public:
    // Data normalization
    static std::vector<double> normalize(const std::vector<double>& data,
                                        const std::string& method = "minmax");
    
    // Data standardization
    static std::vector<double> standardize(const std::vector<double>& data);
    
    // Log transformation
    static std::vector<double> logTransform(const std::vector<double>& data);
    
    // Box-Cox transformation
    static std::vector<double> boxCoxTransform(const std::vector<double>& data,
                                              double lambda);
    
    // Differencing
    static std::vector<double> difference(const std::vector<double>& data,
                                         int order = 1);
    
    // Lag features
    static std::vector<std::vector<double>> createLagFeatures(
        const std::vector<double>& data,
        int max_lag);
};