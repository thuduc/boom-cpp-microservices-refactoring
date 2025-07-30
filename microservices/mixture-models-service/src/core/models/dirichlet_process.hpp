#ifndef DIRICHLET_PROCESS_HPP
#define DIRICHLET_PROCESS_HPP

#include "mixture_base.hpp"
#include <map>

class DirichletProcess : public MixtureModel {
public:
    DirichletProcess();
    
    void setConcentration(double alpha);
    void setBaseMeasure(const nlohmann::json& base_measure);
    
    FitResult fitEM(const Eigen::MatrixXd& data, int max_iterations) override;
    FitResult fitVariational(const Eigen::MatrixXd& data, int max_iterations) override;
    
    std::vector<int> predict(const Eigen::MatrixXd& data) const override;
    Eigen::MatrixXd predictProba(const Eigen::MatrixXd& data) const override;
    std::vector<double> computeDensity(const Eigen::MatrixXd& data) const override;
    
    SampleResult sample(int n_samples, bool return_components) const override;
    
    void setParameters(const nlohmann::json& params) override;
    
private:
    double alpha_;  // Concentration parameter
    nlohmann::json base_measure_;  // Base distribution parameters
    
    // Cluster parameters
    std::map<int, Eigen::VectorXd> cluster_means_;
    std::map<int, Eigen::MatrixXd> cluster_covs_;
    std::map<int, int> cluster_sizes_;
    std::map<int, double> cluster_weights_;
    
    // Chinese Restaurant Process
    std::vector<int> table_assignments_;
    
    void initializeCRP(const Eigen::MatrixXd& data);
    void updateClusters(const Eigen::MatrixXd& data);
    double computePredictiveLikelihood(const Eigen::VectorXd& x, int cluster) const;
    
    // Helper functions
    Eigen::VectorXd drawFromBaseMeasure() const;
    double computeClusterProbability(int cluster_size, int total_customers) const;
};