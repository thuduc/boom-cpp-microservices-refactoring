#include "hierarchical_dp.hpp"
#include <algorithm>
#include <random>
#include <cmath>

HierarchicalDP::HierarchicalDP() : gamma_(1.0), alpha0_(1.0) {
}

void HierarchicalDP::setConcentrations(double gamma, double alpha0) {
    gamma_ = gamma;
    alpha0_ = alpha0;
}

FitResult HierarchicalDP::fitEM(const Eigen::MatrixXd& data, int max_iterations) {
    // For HDP, we need document IDs - for now assume all data is from one document
    std::vector<int> doc_ids(data.rows(), 0);
    int n_docs = 1;
    
    // Initialize HDP structures
    initializeHDP(data, doc_ids);
    
    FitResult result;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Update assignments using Chinese Restaurant Franchise
        for (int d = 0; d < n_docs; ++d) {
            std::vector<int> doc_indices;
            for (int i = 0; i < data.rows(); ++i) {
                if (doc_ids[i] == d) {
                    doc_indices.push_back(i);
                }
            }
            
            for (int idx : doc_indices) {
                // Sample table assignment
                // (Simplified implementation)
            }
        }
        
        // Update global measure
        updateGlobalMeasure();
        
        // Update local measures
        updateLocalMeasures(data, doc_ids);
        
        result.iterations = iter + 1;
    }
    
    // Convert to standard format
    result.n_components = global_atoms_.size();
    result.converged = true;
    
    for (const auto& [k, weight] : global_weights_) {
        result.weights.push_back(weight);
        result.means.push_back(global_atoms_[k]);
        result.covariances.push_back(Eigen::MatrixXd::Identity(data.cols(), data.cols()));
    }
    
    // Compute log likelihood (simplified)
    result.log_likelihood = -data.rows() * std::log(2 * M_PI);
    
    // Approximate BIC/AIC
    int n_params = result.n_components * data.cols();
    result.bic = -2 * result.log_likelihood + n_params * std::log(data.rows());
    result.aic = -2 * result.log_likelihood + 2 * n_params;
    
    return result;
}

FitResult HierarchicalDP::fitVariational(const Eigen::MatrixXd& data, int max_iterations) {
    // Simplified implementation
    return fitEM(data, max_iterations);
}

std::vector<int> HierarchicalDP::predict(const Eigen::MatrixXd& data) const {
    std::vector<int> labels(data.rows());
    
    for (int i = 0; i < data.rows(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        
        int k = 0;
        for (const auto& [cluster_id, atom] : global_atoms_) {
            double dist = (data.row(i).transpose() - atom).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
            k++;
        }
        
        labels[i] = best_cluster;
    }
    
    return labels;
}

Eigen::MatrixXd HierarchicalDP::predictProba(const Eigen::MatrixXd& data) const {
    int n_clusters = global_atoms_.size();
    Eigen::MatrixXd probs(data.rows(), n_clusters);
    
    for (int i = 0; i < data.rows(); ++i) {
        std::vector<double> dists;
        double total = 0.0;
        
        int k = 0;
        for (const auto& [cluster_id, atom] : global_atoms_) {
            double dist = std::exp(-(data.row(i).transpose() - atom).squaredNorm());
            dists.push_back(dist);
            total += dist;
            k++;
        }
        
        for (int j = 0; j < n_clusters; ++j) {
            probs(i, j) = dists[j] / total;
        }
    }
    
    return probs;
}

std::vector<double> HierarchicalDP::computeDensity(const Eigen::MatrixXd& data) const {
    std::vector<double> densities(data.rows());
    
    for (int i = 0; i < data.rows(); ++i) {
        double density = 0.0;
        
        for (const auto& [k, weight] : global_weights_) {
            const Eigen::VectorXd& atom = global_atoms_.at(k);
            double component_density = std::exp(-(data.row(i).transpose() - atom).squaredNorm());
            density += weight * component_density;
        }
        
        densities[i] = density;
    }
    
    return densities;
}

SampleResult HierarchicalDP::sample(int n_samples, bool return_components) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    SampleResult result;
    result.samples = Eigen::MatrixXd(n_samples, global_atoms_.begin()->second.size());
    
    if (return_components) {
        result.component_labels.resize(n_samples);
    }
    
    // Sample from global measure
    std::vector<double> weights;
    std::vector<int> indices;
    
    for (const auto& [k, weight] : global_weights_) {
        weights.push_back(weight);
        indices.push_back(k);
    }
    
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    
    for (int i = 0; i < n_samples; ++i) {
        int idx = dist(gen);
        int k = indices[idx];
        
        if (return_components) {
            result.component_labels[i] = k;
        }
        
        // Add noise to atom
        std::normal_distribution<> noise(0.0, 0.1);
        for (int j = 0; j < result.samples.cols(); ++j) {
            result.samples(i, j) = global_atoms_.at(k)(j) + noise(gen);
        }
    }
    
    return result;
}

void HierarchicalDP::setParameters(const nlohmann::json& params) {
    if (params.contains("gamma")) {
        gamma_ = params["gamma"];
    }
    
    if (params.contains("alpha0")) {
        alpha0_ = params["alpha0"];
    }
    
    if (params.contains("global_measure")) {
        global_weights_.clear();
        global_atoms_.clear();
        
        for (const auto& component : params["global_measure"]) {
            int k = component["index"];
            global_weights_[k] = component["weight"];
            global_atoms_[k] = component["atom"].get<Eigen::VectorXd>();
        }
    }
}

void HierarchicalDP::initializeHDP(const Eigen::MatrixXd& data, const std::vector<int>& doc_ids) {
    // Initialize with simple clustering
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Start with a few global components
    int initial_components = std::min(5, static_cast<int>(data.rows() / 10));
    
    for (int k = 0; k < initial_components; ++k) {
        std::uniform_int_distribution<> dis(0, data.rows() - 1);
        int idx = dis(gen);
        global_atoms_[k] = data.row(idx).transpose();
        global_weights_[k] = 1.0 / initial_components;
    }
    
    // Initialize document-specific structures
    int n_docs = *std::max_element(doc_ids.begin(), doc_ids.end()) + 1;
    doc_weights_.resize(n_docs);
    doc_table_counts_.resize(n_docs);
    table_assignments_.resize(n_docs);
    dish_assignments_.resize(n_docs);
}

void HierarchicalDP::updateGlobalMeasure() {
    // Update global weights using stick-breaking
    double remaining_stick = 1.0;
    
    for (auto& [k, weight] : global_weights_) {
        double beta = computeStickBreaking(k);
        weight = beta * remaining_stick;
        remaining_stick *= (1 - beta);
    }
}

void HierarchicalDP::updateLocalMeasures(const Eigen::MatrixXd& data, 
                                        const std::vector<int>& doc_ids) {
    // Update document-specific measures
    for (int d = 0; d < doc_weights_.size(); ++d) {
        doc_weights_[d].clear();
        
        // Count tables serving each dish
        for (const auto& [dish, count] : doc_table_counts_[d]) {
            doc_weights_[d][dish] = static_cast<double>(count) / table_assignments_[d].size();
        }
    }
}

double HierarchicalDP::computeStickBreaking(int k) const {
    // Simplified stick-breaking
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<> gamma_dist(1.0, gamma_);
    
    return gamma_dist(gen) / (gamma_dist(gen) + gamma_);
}