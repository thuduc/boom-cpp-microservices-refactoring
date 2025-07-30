#include "multivariate_analysis.hpp"
#include "descriptive_stats.hpp"
#include <Eigen/Dense>
#include <algorithm>

PCAResult MultivariateAnalysis::pca(const std::vector<std::vector<double>>& data,
                                    int n_components,
                                    bool standardize) {
    PCAResult result;
    
    // Convert to Eigen matrix
    size_t n_samples = data.size();
    size_t n_features = data[0].size();
    
    Eigen::MatrixXd X(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            X(i, j) = data[i][j];
        }
    }
    
    // Standardize if requested
    if (standardize) {
        Eigen::VectorXd mean = X.colwise().mean();
        Eigen::VectorXd std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() 
                              / (n_samples - 1)).sqrt();
        
        X = (X.rowwise() - mean.transpose()).array().rowwise() / std.transpose().array();
    }
    
    // Compute covariance matrix
    Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = (centered.transpose() * centered) / (n_samples - 1);
    
    // Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();
    
    // Sort by eigenvalues (descending)
    std::vector<std::pair<double, int>> eigen_pairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigen_pairs.push_back({eigenvalues(i), i});
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), std::greater<>());
    
    // Extract top n_components
    n_components = std::min(n_components, static_cast<int>(n_features));
    
    // Explained variance
    double total_variance = eigenvalues.sum();
    double cumulative = 0.0;
    
    for (int i = 0; i < n_components; ++i) {
        double var = eigen_pairs[i].first;
        result.explained_variance.push_back(var);
        result.explained_variance_ratio.push_back(var / total_variance);
        cumulative += var / total_variance;
        result.cumulative_variance_ratio.push_back(cumulative);
    }
    
    // Principal components
    result.components.resize(n_components, std::vector<double>(n_features));
    Eigen::MatrixXd components(n_components, n_features);
    
    for (int i = 0; i < n_components; ++i) {
        int idx = eigen_pairs[i].second;
        for (size_t j = 0; j < n_features; ++j) {
            result.components[i][j] = eigenvectors(j, idx);
            components(i, j) = eigenvectors(j, idx);
        }
    }
    
    // Transform data
    Eigen::MatrixXd transformed = X * components.transpose();
    result.transformed_data.resize(n_samples, std::vector<double>(n_components));
    
    for (size_t i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_components; ++j) {
            result.transformed_data[i][j] = transformed(i, j);
        }
    }
    
    return result;
}

std::vector<std::vector<double>> MultivariateAnalysis::standardizeData(
    const std::vector<std::vector<double>>& data) {
    
    size_t n_samples = data.size();
    size_t n_features = data[0].size();
    
    std::vector<std::vector<double>> standardized(n_samples, 
                                                  std::vector<double>(n_features));
    
    // Compute means and stds for each feature
    for (size_t j = 0; j < n_features; ++j) {
        std::vector<double> column;
        for (size_t i = 0; i < n_samples; ++i) {
            column.push_back(data[i][j]);
        }
        
        double mean = DescriptiveStats::mean(column);
        double std = DescriptiveStats::standardDeviation(column);
        
        for (size_t i = 0; i < n_samples; ++i) {
            standardized[i][j] = (data[i][j] - mean) / std;
        }
    }
    
    return standardized;
}

std::vector<std::vector<double>> MultivariateAnalysis::computeCovarianceMatrix(
    const std::vector<std::vector<double>>& data) {
    
    size_t n_samples = data.size();
    size_t n_features = data[0].size();
    
    std::vector<std::vector<double>> cov(n_features, std::vector<double>(n_features));
    
    // Compute means
    std::vector<double> means(n_features, 0.0);
    for (size_t j = 0; j < n_features; ++j) {
        for (size_t i = 0; i < n_samples; ++i) {
            means[j] += data[i][j];
        }
        means[j] /= n_samples;
    }
    
    // Compute covariance
    for (size_t j1 = 0; j1 < n_features; ++j1) {
        for (size_t j2 = j1; j2 < n_features; ++j2) {
            double sum = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                sum += (data[i][j1] - means[j1]) * (data[i][j2] - means[j2]);
            }
            cov[j1][j2] = cov[j2][j1] = sum / (n_samples - 1);
        }
    }
    
    return cov;
}