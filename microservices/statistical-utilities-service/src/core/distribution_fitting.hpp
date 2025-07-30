#ifndef DISTRIBUTION_FITTING_HPP
#define DISTRIBUTION_FITTING_HPP

#include <vector>
#include <map>
#include <string>

struct DistributionFitResult {
    std::string distribution;
    std::map<std::string, double> parameters;
    double log_likelihood;
    double aic;
    double bic;
    double ks_statistic;
    double ks_p_value;
};

class DistributionFitting {
public:
    static DistributionFitResult fitNormal(const std::vector<double>& data);
    static DistributionFitResult fitExponential(const std::vector<double>& data);
    static DistributionFitResult fitGamma(const std::vector<double>& data);
    static DistributionFitResult fitBeta(const std::vector<double>& data);
    static DistributionFitResult fitWeibull(const std::vector<double>& data);
    
private:
    static double kolmogorovSmirnovTest(const std::vector<double>& data,
                                       const std::string& distribution,
                                       const std::map<std::string, double>& params);
};