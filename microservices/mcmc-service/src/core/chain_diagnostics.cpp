#include "chain_diagnostics.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

double ChainDiagnostics::computeMean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double ChainDiagnostics::computeVariance(const std::vector<double>& data, double mean) {
    if (data.size() <= 1) return 0.0;
    
    double sum_sq = 0.0;
    for (double x : data) {
        sum_sq += (x - mean) * (x - mean);
    }
    return sum_sq / (data.size() - 1);
}

std::vector<double> ChainDiagnostics::computeAutocorrelation(
    const std::vector<double>& data, int max_lag) {
    
    std::vector<double> autocorr(max_lag + 1);
    double mean = computeMean(data);
    double var = computeVariance(data, mean);
    
    if (var == 0) {
        std::fill(autocorr.begin(), autocorr.end(), 1.0);
        return autocorr;
    }
    
    for (int lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (size_t i = 0; i < data.size() - lag; ++i) {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        autocorr[lag] = sum / ((data.size() - lag) * var);
    }
    
    return autocorr;
}

double ChainDiagnostics::computeEffectiveSampleSize(
    const std::vector<double>& data, const std::vector<double>& autocorr) {
    
    // Sum autocorrelations until they become negligible
    double sum_autocorr = 1.0;  // lag 0
    
    for (size_t lag = 1; lag < autocorr.size(); ++lag) {
        if (autocorr[lag] < 0.05) break;  // Threshold
        sum_autocorr += 2 * autocorr[lag];
    }
    
    return data.size() / sum_autocorr;
}

double ChainDiagnostics::computeGewekeZ(const std::vector<double>& data) {
    // Compare means of first 10% and last 50% of chain
    size_t n = data.size();
    size_t n1 = n / 10;
    size_t n2 = n / 2;
    
    if (n1 < 2 || n2 < 2) return 0.0;
    
    // First portion
    std::vector<double> first(data.begin(), data.begin() + n1);
    double mean1 = computeMean(first);
    double var1 = computeVariance(first, mean1);
    
    // Last portion
    std::vector<double> last(data.end() - n2, data.end());
    double mean2 = computeMean(last);
    double var2 = computeVariance(last, mean2);
    
    // Geweke Z-score
    double se = std::sqrt(var1 / n1 + var2 / n2);
    return se > 0 ? (mean1 - mean2) / se : 0.0;
}

DiagnosticResult ChainDiagnostics::compute(
    const std::vector<std::vector<double>>& samples) {
    
    DiagnosticResult result;
    
    if (samples.empty()) return result;
    
    size_t n_samples = samples.size();
    size_t n_params = samples[0].size();
    
    result.mean.resize(n_params);
    result.std_dev.resize(n_params);
    result.effective_sample_size.resize(n_params);
    result.geweke_z.resize(n_params);
    
    // Compute per-parameter diagnostics
    for (size_t param = 0; param < n_params; ++param) {
        // Extract parameter chain
        std::vector<double> param_chain(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            param_chain[i] = samples[i][param];
        }
        
        // Basic statistics
        result.mean[param] = computeMean(param_chain);
        result.std_dev[param] = std::sqrt(computeVariance(param_chain, result.mean[param]));
        
        // Autocorrelation
        int max_lag = std::min(50, static_cast<int>(n_samples / 4));
        auto autocorr = computeAutocorrelation(param_chain, max_lag);
        result.autocorrelation.push_back(autocorr);
        
        // Effective sample size
        result.effective_sample_size[param] = computeEffectiveSampleSize(param_chain, autocorr);
        
        // Geweke diagnostic
        result.geweke_z[param] = computeGewekeZ(param_chain);
        
        // Quantiles
        std::vector<double> sorted_chain = param_chain;
        std::sort(sorted_chain.begin(), sorted_chain.end());
        
        for (double q : {0.025, 0.25, 0.5, 0.75, 0.975}) {
            size_t idx = static_cast<size_t>(q * (n_samples - 1));
            result.quantiles[q].push_back(sorted_chain[idx]);
        }
    }
    
    // Single chain, so no R-hat
    result.gelman_rubin = 1.0;
    
    return result;
}

double ChainDiagnostics::computeRhat(
    const std::vector<std::vector<std::vector<double>>>& chains) {
    
    if (chains.size() < 2) return 1.0;
    
    size_t n_chains = chains.size();
    size_t n_samples = chains[0].size();
    size_t n_params = chains[0][0].size();
    
    // Compute R-hat for first parameter only (simplified)
    std::vector<double> chain_means(n_chains);
    std::vector<double> chain_vars(n_chains);
    
    for (size_t c = 0; c < n_chains; ++c) {
        std::vector<double> param_chain(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            param_chain[i] = chains[c][i][0];
        }
        
        chain_means[c] = computeMean(param_chain);
        chain_vars[c] = computeVariance(param_chain, chain_means[c]);
    }
    
    // Between-chain variance
    double grand_mean = computeMean(chain_means);
    double B = n_samples * computeVariance(chain_means, grand_mean);
    
    // Within-chain variance
    double W = computeMean(chain_vars);
    
    // Potential scale reduction factor
    double var_plus = ((n_samples - 1.0) / n_samples) * W + B / n_samples;
    double rhat = std::sqrt(var_plus / W);
    
    return rhat;
}