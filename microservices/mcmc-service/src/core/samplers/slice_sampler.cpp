#include "../mcmc_engine.hpp"
#include <cmath>
#include <limits>

MCMCResult MCMCEngine::sliceSampler(
    const TargetDistribution& target,
    const std::vector<double>& initial_state,
    const MCMCSettings& settings) {
    
    MCMCResult result;
    result.total_iterations = 0;
    
    std::vector<double> current = initial_state;
    double current_log_density = target(current);
    
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::exponential_distribution<> exponential(1.0);
    
    // Total iterations including burn-in
    int total_iters = settings.burn_in + settings.num_samples * settings.thin;
    
    for (int iter = 0; iter < total_iters; ++iter) {
        // Sample auxiliary variable (slice level)
        double log_y = current_log_density - exponential(getRNG());
        
        // Sample each dimension
        for (size_t dim = 0; dim < current.size(); ++dim) {
            // Create a copy for this dimension
            std::vector<double> x_temp = current;
            
            // Define univariate slice
            auto univariate_target = [&](double x_val) {
                x_temp[dim] = x_val;
                return target(x_temp);
            };
            
            // Find slice interval using stepping out
            double x0 = current[dim];
            double w = settings.slice_width;
            double L = x0 - w * uniform(getRNG());
            double R = L + w;
            
            // Step out left
            int J = settings.max_stepping_out;
            int K = J;
            
            x_temp[dim] = L;
            while (J > 0 && univariate_target(L) > log_y) {
                L -= w;
                x_temp[dim] = L;
                J--;
            }
            
            // Step out right
            x_temp[dim] = R;
            while (K > 0 && univariate_target(R) > log_y) {
                R += w;
                x_temp[dim] = R;
                K--;
            }
            
            // Sample from slice using shrinking
            double x1;
            while (true) {
                x1 = L + uniform(getRNG()) * (R - L);
                x_temp[dim] = x1;
                
                if (univariate_target(x1) > log_y) {
                    break;  // Accept
                }
                
                // Shrink interval
                if (x1 < x0) {
                    L = x1;
                } else {
                    R = x1;
                }
                
                // Safety check
                if (R - L < 1e-10) {
                    x1 = (L + R) / 2;
                    break;
                }
            }
            
            current[dim] = x1;
        }
        
        // Update log density
        current_log_density = target(current);
        
        // Store sample if past burn-in and at thinning interval
        if (iter >= settings.burn_in && (iter - settings.burn_in) % settings.thin == 0) {
            result.samples.push_back(current);
            result.log_densities.push_back(current_log_density);
        }
        
        result.total_iterations = iter + 1;
    }
    
    // Slice sampler always accepts
    result.acceptance_rate = 1.0;
    
    // Compute effective sample size (placeholder)
    result.effective_sample_size.resize(current.size());
    for (size_t i = 0; i < current.size(); ++i) {
        result.effective_sample_size[i] = result.samples.size();
    }
    
    return result;
}