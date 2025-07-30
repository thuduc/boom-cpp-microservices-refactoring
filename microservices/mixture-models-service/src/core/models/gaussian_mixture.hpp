#ifndef GAUSSIAN_MIXTURE_HPP
#define GAUSSIAN_MIXTURE_HPP

#include "mixture_base.hpp"
#include <random>

class GaussianMixture : public MixtureModel {
public:
    GaussianMixture(int n_components = 3);
    
    void setNumComponents(int k) { n_components_ = k; }
    void setCovarianceType(const std::string& type) { covariance_type_ = type; }
    
    // MixtureModel interface
    FitResult fitEM(const Eigen::MatrixXd& data, int max_iter = 100) override;
    FitResult fitVariational(const Eigen::MatrixXd& data, int max_iter = 100) override;
    
    std::vector<int> predict(const Eigen::MatrixXd& data) override;
    Eigen::MatrixXd predictProba(const Eigen::MatrixXd& data) override;
    std::vector<double> computeDensity(const Eigen::MatrixXd& data) override;
    
    SampleResult sample(int n_samples, bool return_components = false) override;
    
    void setParameters(const json& params) override;
    json getParameters() const override;
    
private:
    int n_components_;
    std::string covariance_type_;  // "full", "tied", "diag", "spherical"
    
    std::vector<double> weights_;
    std::vector<Eigen::VectorXd> means_;
    std::vector<Eigen::MatrixXd> covariances_;
    
    mutable std::mt19937 rng_;
    
    // EM algorithm steps
    void initialize(const Eigen::MatrixXd& data);
    Eigen::MatrixXd eStep(const Eigen::MatrixXd& data);
    void mStep(const Eigen::MatrixXd& data, const Eigen::MatrixXd& responsibilities);
    double computeLogLikelihood(const Eigen::MatrixXd& data) const;
    
    // Multivariate Gaussian PDF
    double gaussianPdf(const Eigen::VectorXd& x, 
                      const Eigen::VectorXd& mean,
                      const Eigen::MatrixXd& cov) const;
    
    // Covariance constraints
    Eigen::MatrixXd constrainCovariance(const Eigen::MatrixXd& cov) const;
};

#endif // GAUSSIAN_MIXTURE_HPP