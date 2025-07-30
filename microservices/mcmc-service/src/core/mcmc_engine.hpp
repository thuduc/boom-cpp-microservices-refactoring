#ifndef MCMC_ENGINE_HPP
#define MCMC_ENGINE_HPP

#include <vector>
#include <functional>
#include <random>

// Type definition for target distribution (log density)
using TargetDistribution = std::function<double(const std::vector<double>&)>;

struct MCMCSettings {
    int num_samples = 1000;
    int burn_in = 100;
    int thin = 1;
    
    // Metropolis-Hastings settings
    std::string proposal_type = "normal";
    double proposal_scale = 1.0;
    int adaptation_period = 500;
    
    // Slice sampler settings
    double slice_width = 1.0;
    int max_stepping_out = 10;
    
    // ARMS settings
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    
    // Gibbs settings
    bool random_scan = false;  // vs systematic scan
};

struct MCMCResult {
    std::vector<std::vector<double>> samples;
    double acceptance_rate;
    std::vector<double> effective_sample_size;
    std::vector<double> log_densities;
    int total_iterations;
};

class MCMCEngine {
public:
    // Metropolis-Hastings sampler
    static MCMCResult metropolisHastings(
        const TargetDistribution& target,
        const std::vector<double>& initial_state,
        const MCMCSettings& settings);
    
    // Slice sampler
    static MCMCResult sliceSampler(
        const TargetDistribution& target,
        const std::vector<double>& initial_state,
        const MCMCSettings& settings);
    
    // Adaptive Rejection Metropolis Sampling
    static MCMCResult arms(
        const TargetDistribution& target,
        const std::vector<double>& initial_state,
        const MCMCSettings& settings);
    
    // Gibbs sampler
    static MCMCResult gibbs(
        const std::vector<TargetDistribution>& conditionals,
        const std::vector<double>& initial_state,
        const MCMCSettings& settings);
    
private:
    // Helper functions
    static std::mt19937& getRNG();
    
    // Proposal distributions
    static std::vector<double> normalProposal(
        const std::vector<double>& current,
        double scale);
    
    static std::vector<double> uniformProposal(
        const std::vector<double>& current,
        double scale);
};

#endif // MCMC_ENGINE_HPP