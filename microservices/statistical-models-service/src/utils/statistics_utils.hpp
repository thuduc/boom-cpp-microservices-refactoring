#ifndef STATISTICS_UTILS_HPP
#define STATISTICS_UTILS_HPP

#include <vector>

class StatisticsUtils {
public:
    static double mean(const std::vector<double>& data);
    static double variance(const std::vector<double>& data);
    static double standardDeviation(const std::vector<double>& data);
    static double quantile(const std::vector<double>& data, double p);
};

#endif // STATISTICS_UTILS_HPP