#ifndef HIERARCHICAL_DP_HPP
#define HIERARCHICAL_DP_HPP

#include "mixture_base.hpp"
#include <map>
#include <vector>

class HierarchicalDP : public MixtureModel {
public:
    HierarchicalDP();
    
    void setConcentrations(double gamma, double alpha0);
    
    FitResult fitEM(const Eigen::MatrixXd& data, int max_iterations) override;
    FitResult fitVariational(const Eigen::MatrixXd& data, int max_iterations) override;
    
    std::vector<int> predict(const Eigen::MatrixXd& data) const override;
    Eigen::MatrixXd predictProba(const Eigen::MatrixXd& data) const override;
    std::vector<double> computeDensity(const Eigen::MatrixXd& data) const override;
    
    SampleResult sample(int n_samples, bool return_components) const override;
    
    void setParameters(const nlohmann::json& params) override;
    
private:
    double gamma_;   // Top-level concentration
    double alpha0_;  // Document-level concentration
    
    // Global measure
    std::map<int, double> global_weights_;
    std::map<int, Eigen::VectorXd> global_atoms_;
    
    // Document-specific measures
    std::vector<std::map<int, double>> doc_weights_;
    std::vector<std::map<int, int>> doc_table_counts_;
    
    // Assignments
    std::vector<std::vector<int>> table_assignments_;
    std::vector<std::vector<int>> dish_assignments_;
    
    void initializeHDP(const Eigen::MatrixXd& data, const std::vector<int>& doc_ids);
    void updateGlobalMeasure();
    void updateLocalMeasures(const Eigen::MatrixXd& data, const std::vector<int>& doc_ids);
    
    double computeStickBreaking(int k) const;
};