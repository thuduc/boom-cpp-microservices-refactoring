#ifndef DESCRIPTIVE_STATS_HPP
#define DESCRIPTIVE_STATS_HPP

#include <vector>
#include <map>

struct Spline {
    double evaluate(double x) const;
    std::vector<double> getKnots() const;
    // Implementation details...
    std::vector<double> knots;
    std::vector<double> coefficients;
};

class DescriptiveStats {
public:
    // Central tendency
    static double mean(const std::vector<double>& data);
    static double median(std::vector<double> data);  // Copy for sorting
    static std::vector<double> mode(const std::vector<double>& data);
    
    // Dispersion
    static double variance(const std::vector<double>& data, bool sample = true);
    static double standardDeviation(const std::vector<double>& data, bool sample = true);
    static double range(const std::vector<double>& data);
    static double interquartileRange(const std::vector<double>& data);
    
    // Shape
    static double skewness(const std::vector<double>& data);
    static double kurtosis(const std::vector<double>& data);
    
    // Position
    static double min(const std::vector<double>& data);
    static double max(const std::vector<double>& data);
    static std::vector<double> quantiles(const std::vector<double>& data, 
                                        const std::vector<double>& probs);
    
    // ECDF
    static std::vector<double> ecdf(const std::vector<double>& data,
                                   const std::vector<double>& eval_points);
    
    // Spline fitting
    static Spline fitSpline(const std::vector<double>& x,
                           const std::vector<double>& y,
                           int degree = 3,
                           double smoothing = 0.0);
    
private:
    static double moment(const std::vector<double>& data, int order, double center);
};