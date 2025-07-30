#include "dirichlet_process.hpp"
#include <algorithm>
#include <random>
#include <cmath>

DirichletProcess::DirichletProcess() : alpha_(1.0) {
    // Default base measure: Normal-Inverse-Wishart
    base_measure_ = {
        {"type", "niw"},
        {"mu0", std::vector<double>{0.0, 0.0}},
        {"kappa0", 0.01},
        {"nu0", 4},
        {"lambda0", std::vector<std::vector<double>>{{1.0, 0.0}, {0.0, 1.0}}}
    };
}

void DirichletProcess::setConcentration(double alpha) {
    alpha_ = alpha;
}

void DirichletProcess::setBaseMeasure(const nlohmann::json& base_measure) {
    base_measure_ = base_measure;
}

FitResult DirichletProcess::fitEM(const Eigen::MatrixXd& data, int max_iterations) {
    // For DP, we use a Gibbs sampling approach rather than standard EM
    int n_samples = data.rows();
    
    // Initialize with Chinese Restaurant Process
    initializeCRP(data);
    
    FitResult result;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // For each data point, reassign to clusters
        for (int i = 0; i < n_samples; ++i) {
            // Remove current assignment
            int current_table = table_assignments_[i];
            cluster_sizes_[current_table]--;
            if (cluster_sizes_[current_table] == 0) {
                cluster_means_.erase(current_table);
                cluster_covs_.erase(current_table);
                cluster_sizes_.erase(current_table);
            }
            
            // Compute probabilities for each existing cluster + new cluster
            std::vector<double> probs;
            std::vector<int> cluster_ids;
            
            // Existing clusters
            for (const auto& [cluster_id, size] : cluster_sizes_) {
                double prob = size * computePredictiveLikelihood(data.row(i), cluster_id);
                probs.push_back(prob);
                cluster_ids.push_back(cluster_id);
            }
            
            // New cluster
            double new_cluster_prob = alpha_ * computePredictiveLikelihood(data.row(i), -1);
            probs.push_back(new_cluster_prob);
            cluster_ids.push_back(cluster_ids.empty() ? 0 : cluster_ids.back() + 1);
            
            // Sample new assignment
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            int selected_idx = dist(gen);
            int new_cluster = cluster_ids[selected_idx];
            
            // Update assignment
            table_assignments_[i] = new_cluster;
            cluster_sizes_[new_cluster]++;
            
            // Update cluster parameters if needed
            if (cluster_means_.find(new_cluster) == cluster_means_.end()) {
                // Initialize new cluster
                cluster_means_[new_cluster] = drawFromBaseMeasure();
                cluster_covs_[new_cluster] = Eigen::MatrixXd::Identity(data.cols(), data.cols());
            }
        }
        
        // Update cluster parameters
        updateClusters(data);
        
        result.iterations = iter + 1;
    }
    
    // Convert to standard format
    result.n_components = cluster_means_.size();
    result.converged = true;
    
    // Compute weights
    int total = std::accumulate(cluster_sizes_.begin(), cluster_sizes_.end(), 0,
                               [](int sum, const auto& p) { return sum + p.second; });
    
    for (const auto& [cluster_id, size] : cluster_sizes_) {
        result.weights.push_back(static_cast<double>(size) / total);
        result.means.push_back(cluster_means_[cluster_id]);
        result.covariances.push_back(cluster_covs_[cluster_id]);
    }
    
    // Compute log likelihood
    result.log_likelihood = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        int cluster = table_assignments_[i];
        result.log_likelihood += std::log(computePredictiveLikelihood(data.row(i), cluster));
    }
    
    // Approximate BIC/AIC
    int n_params = result.n_components * (data.cols() + data.cols() * (data.cols() + 1) / 2);
    result.bic = -2 * result.log_likelihood + n_params * std::log(n_samples);
    result.aic = -2 * result.log_likelihood + 2 * n_params;
    
    return result;
}

FitResult DirichletProcess::fitVariational(const Eigen::MatrixXd& data, int max_iterations) {
    // Simplified implementation - delegate to Gibbs sampling
    return fitEM(data, max_iterations);
}

std::vector<int> DirichletProcess::predict(const Eigen::MatrixXd& data) const {
    std::vector<int> labels(data.rows());
    
    for (int i = 0; i < data.rows(); ++i) {
        double max_prob = -std::numeric_limits<double>::infinity();
        int best_cluster = 0;
        
        for (const auto& [cluster_id, mean] : cluster_means_) {
            double prob = computePredictiveLikelihood(data.row(i), cluster_id);
            if (prob > max_prob) {
                max_prob = prob;
                best_cluster = cluster_id;
            }
        }
        
        labels[i] = best_cluster;
    }
    
    return labels;
}

Eigen::MatrixXd DirichletProcess::predictProba(const Eigen::MatrixXd& data) const {
    int n_clusters = cluster_means_.size();
    Eigen::MatrixXd probs(data.rows(), n_clusters);
    
    std::vector<int> cluster_indices;
    for (const auto& [cluster_id, _] : cluster_means_) {
        cluster_indices.push_back(cluster_id);
    }
    
    for (int i = 0; i < data.rows(); ++i) {
        double total_prob = 0.0;
        
        for (int j = 0; j < n_clusters; ++j) {
            int cluster_id = cluster_indices[j];
            double prob = cluster_weights_.at(cluster_id) * 
                         computePredictiveLikelihood(data.row(i), cluster_id);
            probs(i, j) = prob;
            total_prob += prob;
        }
        
        // Normalize
        if (total_prob > 0) {
            probs.row(i) /= total_prob;
        }
    }
    
    return probs;
}

std::vector<double> DirichletProcess::computeDensity(const Eigen::MatrixXd& data) const {
    std::vector<double> densities(data.rows());
    
    for (int i = 0; i < data.rows(); ++i) {
        double density = 0.0;
        
        for (const auto& [cluster_id, weight] : cluster_weights_) {
            density += weight * computePredictiveLikelihood(data.row(i), cluster_id);
        }
        
        densities[i] = density;
    }
    
    return densities;
}

SampleResult DirichletProcess::sample(int n_samples, bool return_components) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    SampleResult result;
    result.samples = Eigen::MatrixXd(n_samples, cluster_means_.begin()->second.size());
    
    if (return_components) {
        result.component_labels.resize(n_samples);
    }
    
    // Sample from stick-breaking representation
    std::vector<double> weights;
    std::vector<int> cluster_ids;
    
    for (const auto& [cluster_id, weight] : cluster_weights_) {
        weights.push_back(weight);
        cluster_ids.push_back(cluster_id);
    }
    
    std::discrete_distribution<> cluster_dist(weights.begin(), weights.end());
    
    for (int i = 0; i < n_samples; ++i) {
        int idx = cluster_dist(gen);
        int cluster_id = cluster_ids[idx];
        
        if (return_components) {
            result.component_labels[i] = cluster_id;
        }
        
        // Sample from multivariate normal
        std::normal_distribution<> dist(0.0, 1.0);
        Eigen::VectorXd z(result.samples.cols());
        for (int j = 0; j < z.size(); ++j) {
            z(j) = dist(gen);
        }
        
        Eigen::LLT<Eigen::MatrixXd> llt(cluster_covs_.at(cluster_id));
        result.samples.row(i) = (cluster_means_.at(cluster_id) + llt.matrixL() * z).transpose();
    }
    
    return result;
}

void DirichletProcess::setParameters(const nlohmann::json& params) {
    if (params.contains("alpha")) {
        alpha_ = params["alpha"];
    }
    
    if (params.contains("clusters")) {
        cluster_means_.clear();
        cluster_covs_.clear();
        cluster_weights_.clear();
        
        for (const auto& cluster : params["clusters"]) {
            int id = cluster["id"];
            cluster_means_[id] = cluster["mean"].get<Eigen::VectorXd>();
            cluster_covs_[id] = cluster["covariance"].get<Eigen::MatrixXd>();
            cluster_weights_[id] = cluster["weight"];
        }
    }
}

void DirichletProcess::initializeCRP(const Eigen::MatrixXd& data) {
    int n_samples = data.rows();
    table_assignments_.resize(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // First customer sits at first table
    table_assignments_[0] = 0;
    cluster_sizes_[0] = 1;
    cluster_means_[0] = data.row(0).transpose();
    cluster_covs_[0] = Eigen::MatrixXd::Identity(data.cols(), data.cols());
    
    // Subsequent customers follow CRP
    for (int i = 1; i < n_samples; ++i) {
        std::vector<double> probs;
        std::vector<int> tables;
        
        // Existing tables
        for (const auto& [table, size] : cluster_sizes_) {
            probs.push_back(size);
            tables.push_back(table);
        }
        
        // New table
        probs.push_back(alpha_);
        tables.push_back(tables.empty() ? 0 : tables.back() + 1);
        
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int selected_idx = dist(gen);
        int table = tables[selected_idx];
        
        table_assignments_[i] = table;
        cluster_sizes_[table]++;
        
        if (selected_idx == tables.size() - 1) {
            // New table
            cluster_means_[table] = data.row(i).transpose();
            cluster_covs_[table] = Eigen::MatrixXd::Identity(data.cols(), data.cols());
        }
    }
    
    updateClusters(data);
}

void DirichletProcess::updateClusters(const Eigen::MatrixXd& data) {
    // Update cluster parameters based on assigned data points
    for (auto& [cluster_id, mean] : cluster_means_) {
        std::vector<int> cluster_points;
        for (int i = 0; i < table_assignments_.size(); ++i) {
            if (table_assignments_[i] == cluster_id) {
                cluster_points.push_back(i);
            }
        }
        
        if (!cluster_points.empty()) {
            // Update mean
            Eigen::VectorXd new_mean = Eigen::VectorXd::Zero(data.cols());
            for (int idx : cluster_points) {
                new_mean += data.row(idx).transpose();
            }
            mean = new_mean / cluster_points.size();
            
            // Update covariance
            Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(data.cols(), data.cols());
            for (int idx : cluster_points) {
                Eigen::VectorXd diff = data.row(idx).transpose() - mean;
                cov += diff * diff.transpose();
            }
            cluster_covs_[cluster_id] = cov / cluster_points.size() + 
                                       Eigen::MatrixXd::Identity(data.cols(), data.cols()) * 1e-6;
        }
    }
    
    // Update weights
    int total = table_assignments_.size();
    for (const auto& [cluster_id, size] : cluster_sizes_) {
        cluster_weights_[cluster_id] = static_cast<double>(size) / total;
    }
}

double DirichletProcess::computePredictiveLikelihood(const Eigen::VectorXd& x, int cluster) const {
    if (cluster == -1 || cluster_means_.find(cluster) == cluster_means_.end()) {
        // New cluster - use base measure
        Eigen::VectorXd base_mean = drawFromBaseMeasure();
        Eigen::MatrixXd base_cov = Eigen::MatrixXd::Identity(x.size(), x.size());
        
        // Compute Gaussian likelihood
        Eigen::VectorXd diff = x - base_mean;
        double det = base_cov.determinant();
        Eigen::MatrixXd inv_cov = base_cov.inverse();
        
        double exponent = -0.5 * diff.transpose() * inv_cov * diff;
        double normalization = std::pow(2 * M_PI, -x.size()/2.0) * std::pow(det, -0.5);
        
        return normalization * std::exp(exponent);
    } else {
        // Existing cluster
        const Eigen::VectorXd& mean = cluster_means_.at(cluster);
        const Eigen::MatrixXd& cov = cluster_covs_.at(cluster);
        
        Eigen::VectorXd diff = x - mean;
        double det = cov.determinant();
        if (det <= 0) return 1e-10;
        
        Eigen::MatrixXd inv_cov = cov.inverse();
        double exponent = -0.5 * diff.transpose() * inv_cov * diff;
        double normalization = std::pow(2 * M_PI, -x.size()/2.0) * std::pow(det, -0.5);
        
        return normalization * std::exp(exponent);
    }
}

Eigen::VectorXd DirichletProcess::drawFromBaseMeasure() const {
    // Simple implementation - draw from prior
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);
    
    if (base_measure_["type"] == "niw") {
        auto mu0 = base_measure_["mu0"].get<std::vector<double>>();
        Eigen::VectorXd mean(mu0.size());
        
        for (int i = 0; i < mu0.size(); ++i) {
            mean(i) = mu0[i] + dist(gen);
        }
        
        return mean;
    }
    
    // Default
    Eigen::VectorXd mean(2);
    mean(0) = dist(gen);
    mean(1) = dist(gen);
    return mean;
}

double DirichletProcess::computeClusterProbability(int cluster_size, int total_customers) const {
    if (cluster_size == 0) {
        return alpha_ / (total_customers + alpha_);
    } else {
        return cluster_size / (total_customers + alpha_);
    }
}