#ifndef TIME_SERIES_ANALYSIS_HPP
#define TIME_SERIES_ANALYSIS_HPP

#include <vector>

struct TrendResult {
    double slope;
    double intercept;
    double r_squared;
    std::vector<double> detrended;
};

struct SeasonalResult {
    std::vector<double> seasonal;
    std::vector<double> trend;
    std::vector<double> residual;
};

struct StationarityResult {
    double statistic;
    double p_value;
    bool is_stationary;
};

class TimeSeriesAnalysis {
public:
    static std::vector<double> autocorrelation(const std::vector<double>& data, int max_lag);
    static std::vector<double> partialAutocorrelation(const std::vector<double>& data, int max_lag);
    static TrendResult detrend(const std::vector<double>& data);
    static SeasonalResult seasonalDecomposition(const std::vector<double>& data, int period);
    static StationarityResult augmentedDickeyFuller(const std::vector<double>& data);
    
private:
    static double computeACF(const std::vector<double>& data, int lag);
};