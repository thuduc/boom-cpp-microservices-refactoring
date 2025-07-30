#ifndef RESAMPLING_METHODS_HPP
#define RESAMPLING_METHODS_HPP

#include <vector>
#include <string>

struct ResamplingResult {
    double estimate;
    double standard_error;
    double bias;
    std::vector<double> confidence_interval;
    std::vector<double> samples;
};

class ResamplingMethods {
public:
    // Bootstrap methods
    static ResamplingResult bootstrap(const std::vector<double>& data,
                                     int n_samples,
                                     const std::string& statistic = "mean");
    
    // Jackknife resampling
    static ResamplingResult jackknife(const std::vector<double>& data,
                                     const std::string& statistic = "mean");
    
    // Permutation test
    static ResamplingResult permutationTest(const std::vector<double>& data1,
                                           const std::vector<double>& data2,
                                           int n_permutations,
                                           const std::string& statistic = "mean_diff");
    
    // Cross-validation
    static std::vector<double> crossValidation(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              int n_folds);
    
private:
    // Helper to compute various statistics
    static double computeStatistic(const std::vector<double>& data,
                                  const std::string& statistic);
    
    static double computeStatistic(const std::vector<double>& data1,
                                  const std::vector<double>& data2,
                                  const std::string& statistic);
};