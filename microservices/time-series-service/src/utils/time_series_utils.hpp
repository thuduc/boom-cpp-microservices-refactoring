#ifndef TIME_SERIES_UTILS_HPP
#define TIME_SERIES_UTILS_HPP

#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct DecompositionResult {
    std::vector<double> trend;
    std::vector<double> seasonal;
    std::vector<double> remainder;
};

struct TestResult {
    double statistic;
    double p_value;
    int df;
};

class TimeSeriesUtils {
public:
    // Data conversion
    static std::vector<double> jsonToVector(const json& data);
    static json vectorToJson(const std::vector<double>& vec);
    
    // Basic statistics
    static double mean(const std::vector<double>& data);
    static double variance(const std::vector<double>& data);
    static double skewness(const std::vector<double>& data);
    static double kurtosis(const std::vector<double>& data);
    
    // Autocorrelation functions
    static std::vector<double> autocorrelation(const std::vector<double>& data, int max_lag);
    static std::vector<double> partialAutocorrelation(const std::vector<double>& data, int max_lag);
    
    // Statistical tests
    static TestResult ljungBoxTest(const std::vector<double>& residuals, int lags);
    static TestResult augmentedDickeyFullerTest(const std::vector<double>& data);
    
    // Decomposition methods
    static DecompositionResult stlDecomposition(const std::vector<double>& data, int period);
    static DecompositionResult classicalDecomposition(const std::vector<double>& data, 
                                                      int period, 
                                                      bool multiplicative = false);
    
    // Transformations
    static std::vector<double> boxCoxTransform(const std::vector<double>& data, double lambda);
    static std::vector<double> inverseBoxCoxTransform(const std::vector<double>& data, double lambda);
    static double findOptimalBoxCoxLambda(const std::vector<double>& data);
    
    // Moving averages
    static std::vector<double> movingAverage(const std::vector<double>& data, int window);
    static std::vector<double> exponentialMovingAverage(const std::vector<double>& data, double alpha);
    
private:
    // Helper for STL decomposition
    static std::vector<double> loess(const std::vector<double>& x, 
                                    const std::vector<double>& y, 
                                    double bandwidth);
};

#endif // TIME_SERIES_UTILS_HPP