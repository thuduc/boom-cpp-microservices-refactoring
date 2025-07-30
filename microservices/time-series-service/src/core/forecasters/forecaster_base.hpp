#ifndef FORECASTER_BASE_HPP
#define FORECASTER_BASE_HPP

#include <vector>

class ForecasterBase {
public:
    virtual ~ForecasterBase() = default;
    
    // Simple exponential smoothing
    static std::vector<double> exponentialSmoothing(
        const std::vector<double>& data,
        double alpha,
        int horizon);
    
    // Holt's linear trend method
    static std::vector<double> holtsMethod(
        const std::vector<double>& data,
        double alpha,
        double beta,
        int horizon);
    
    // Holt-Winters seasonal method
    static std::vector<double> holtWinters(
        const std::vector<double>& data,
        double alpha,
        double beta,
        double gamma,
        int period,
        int horizon,
        bool multiplicative = false);
};

#endif // FORECASTER_BASE_HPP