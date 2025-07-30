#ifndef CHAIN_DIAGNOSTICS_HPP
#define CHAIN_DIAGNOSTICS_HPP

#include <vector>
#include <map>

struct DiagnosticResult {
    std::vector<double> mean;
    std::vector<double> std_dev;
    std::map<double, std::vector<double>> quantiles;  // quantile -> values
    std::vector<std::vector<double>> autocorrelation;  // lag -> correlation
    std::vector<double> effective_sample_size;
    double gelman_rubin;  // R-hat statistic (if multiple chains)
    std::vector<double> geweke_z;  // Geweke convergence diagnostic
};

class ChainDiagnostics {
public:
    // Compute comprehensive diagnostics for MCMC chain
    DiagnosticResult compute(const std::vector<std::vector<double>>& samples);
    
    // Compute R-hat (Gelman-Rubin) statistic for multiple chains
    static double computeRhat(const std::vector<std::vector<std::vector<double>>>& chains);
    
private:
    // Helper functions
    static double computeMean(const std::vector<double>& data);
    static double computeVariance(const std::vector<double>& data, double mean);
    static std::vector<double> computeAutocorrelation(const std::vector<double>& data, int max_lag);
    static double computeEffectiveSampleSize(const std::vector<double>& data, 
                                           const std::vector<double>& autocorr);
    static double computeGewekeZ(const std::vector<double>& data);
};

#endif // CHAIN_DIAGNOSTICS_HPP