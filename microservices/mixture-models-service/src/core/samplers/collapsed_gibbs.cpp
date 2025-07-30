#include "collapsed_gibbs.hpp"
#include <random>
#include <cmath>

FitResult CollapsedGibbs::fit(MixtureModel* model, const Eigen::MatrixXd& data, int n_iterations) {
    FitResult result;
    int n_samples = data.rows();
    
    // Initialize with Chinese Restaurant Process
    std::vector<int> assignments(n_samples);
    std::vector<int> cluster_sizes;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // First customer
    assignments[0] = 0;
    cluster_sizes.push_back(1);
    
    // Subsequent customers
    for (int i = 1; i < n_samples; ++i) {
        std::vector<double> probs;
        
        // Existing tables
        for (int k = 0; k < cluster_sizes.size(); ++k) {
            probs.push_back(cluster_sizes[k]);
        }
        
        // New table (with concentration parameter alpha = 1.0)
        probs.push_back(1.0);
        
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int table = dist(gen);
        
        assignments[i] = table;
        if (table == cluster_sizes.size()) {
            cluster_sizes.push_back(1);
        } else {
            cluster_sizes[table]++;
        }
    }
    
    // Collapsed Gibbs sampling
    for (int iter = 0; iter < n_iterations; ++iter) {
        for (int i = 0; i < n_samples; ++i) {
            // Remove from current cluster
            int current_cluster = assignments[i];
            cluster_sizes[current_cluster]--;
            
            // Compute probabilities for each cluster
            std::vector<double> probs;
            
            for (int k = 0; k < cluster_sizes.size(); ++k) {
                if (cluster_sizes[k] == 0 && k != current_cluster) continue;
                
                // Collect data points in cluster k (excluding current point)
                Eigen::MatrixXd cluster_data(cluster_sizes[k] + 1, data.cols());
                int idx = 0;
                for (int j = 0; j < n_samples; ++j) {
                    if (j != i && assignments[j] == k) {
                        cluster_data.row(idx++) = data.row(j);
                    }
                }
                cluster_data.row(idx) = data.row(i);
                
                double marginal = computeMarginalLikelihood(cluster_data.topRows(idx + 1));
                probs.push_back(cluster_sizes[k] * marginal);
            }
            
            // New cluster
            double new_marginal = computeMarginalLikelihood(data.row(i));
            probs.push_back(1.0 * new_marginal);
            
            // Sample new assignment
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            int new_cluster = dist(gen);
            
            if (new_cluster == probs.size() - 1) {
                // Create new cluster
                assignments[i] = cluster_sizes.size();
                cluster_sizes.push_back(1);
            } else {
                assignments[i] = new_cluster;
                cluster_sizes[new_cluster]++;
            }
            
            // Clean up empty clusters
            // (simplified - in practice would need to renumber)
        }
        
        result.iterations = iter + 1;
    }
    
    result.converged = true;
    result.n_components = *std::max_element(cluster_sizes.begin(), cluster_sizes.end()) + 1;
    
    return result;
}

double CollapsedGibbs::computeMarginalLikelihood(const Eigen::MatrixXd& cluster_data) const {
    // Simplified marginal likelihood computation
    // In practice, this would compute the marginal likelihood
    // after integrating out the parameters
    
    int n = cluster_data.rows();
    int d = cluster_data.cols();
    
    if (n == 0) return 1e-10;
    
    // Use Normal-Inverse-Wishart conjugate prior
    // Simplified computation
    double log_marginal = -0.5 * n * d * std::log(2 * M_PI);
    
    // Add data-dependent terms (simplified)
    Eigen::VectorXd mean = cluster_data.colwise().mean();
    Eigen::MatrixXd centered = cluster_data.rowwise() - mean.transpose();
    double scatter = centered.squaredNorm();
    
    log_marginal -= 0.5 * scatter / (1.0 + n);
    
    return std::exp(log_marginal);
}