#ifndef MULTIVARIATE_ANALYSIS_HPP
#define MULTIVARIATE_ANALYSIS_HPP

#include <vector>

struct PCAResult {
    std::vector<double> explained_variance;
    std::vector<double> explained_variance_ratio;
    std::vector<double> cumulative_variance_ratio;
    std::vector<std::vector<double>> components;
    std::vector<std::vector<double>> transformed_data;
};

class MultivariateAnalysis {
public:
    static PCAResult pca(const std::vector<std::vector<double>>& data,
                        int n_components,
                        bool standardize = true);
    
private:
    static std::vector<std::vector<double>> standardizeData(
        const std::vector<std::vector<double>>& data);
    
    static std::vector<std::vector<double>> computeCovarianceMatrix(
        const std::vector<std::vector<double>>& data);
};