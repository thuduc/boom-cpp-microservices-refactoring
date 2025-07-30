#include "../mcmc_engine.hpp"
#include <cmath>
#include <algorithm>

MCMCResult MCMCEngine::metropolisHastings(
    const TargetDistribution& target,
    const std::vector<double>& initial_state,
    const MCMCSettings& settings) {
    
    MCMCResult result;
    result.total_iterations = 0;
    
    std::vector<double> current = initial_state;
    double current_log_density = target(current);
    
    int accepted = 0;
    double proposal_scale = settings.proposal_scale;
    
    // Adaptive scaling
    std::vector<int> adaptation_accepts(100, 0);
    int adaptation_index = 0;
    
    // Total iterations including burn-in
    int total_iters = settings.burn_in + settings.num_samples * settings.thin;
    
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    
    for (int iter = 0; iter < total_iters; ++iter) {
        // Propose new state
        std::vector<double> proposed;
        if (settings.proposal_type == "normal") {
            proposed = normalProposal(current, proposal_scale);
        } else {
            proposed = uniformProposal(current, proposal_scale);
        }
        
        // Compute acceptance probability
        double proposed_log_density = target(proposed);
        double log_acceptance_prob = proposed_log_density - current_log_density;
        
        // Accept or reject
        bool accept = false;
        if (log_acceptance_prob >= 0) {
            accept = true;
        } else {
            double u = uniform(getRNG());
            if (std::log(u) < log_acceptance_prob) {
                accept = true;
            }
        }
        
        if (accept) {
            current = proposed;
            current_log_density = proposed_log_density;
            accepted++;
            
            if (iter < settings.adaptation_period) {
                adaptation_accepts[adaptation_index] = 1;
            }
        } else {
            if (iter < settings.adaptation_period) {
                adaptation_accepts[adaptation_index] = 0;
            }
        }
        
        // Adaptation
        if (iter < settings.adaptation_period) {
            adaptation_index = (adaptation_index + 1) % 100;
            
            if (iter > 0 && iter % 100 == 0) {
                // Compute acceptance rate over last 100 iterations
                double recent_acceptance = std::accumulate(
                    adaptation_accepts.begin(), adaptation_accepts.end(), 0.0) / 100.0;
                
                // Adjust proposal scale to target ~0.234 acceptance rate
                if (recent_acceptance > 0.3) {
                    proposal_scale *= 1.1;
                } else if (recent_acceptance < 0.15) {
                    proposal_scale *= 0.9;
                }
            }
        }
        
        // Store sample if past burn-in and at thinning interval
        if (iter >= settings.burn_in && (iter - settings.burn_in) % settings.thin == 0) {
            result.samples.push_back(current);
            result.log_densities.push_back(current_log_density);
        }
        
        result.total_iterations = iter + 1;
    }
    
    result.acceptance_rate = static_cast<double>(accepted) / total_iters;
    
    // Compute effective sample size (placeholder - would need autocorrelation)
    result.effective_sample_size.resize(current.size());
    for (size_t i = 0; i < current.size(); ++i) {
        result.effective_sample_size[i] = result.samples.size() * result.acceptance_rate;
    }
    
    return result;
}