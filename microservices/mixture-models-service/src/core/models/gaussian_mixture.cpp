#include "gaussian_mixture.hpp"
#include <cmath>
#include <algorithm>
#include <random>

GaussianMixture::GaussianMixture() 
    : n_components_(3), covariance_type_("full"), reg_covar_(1e-6) {
}

void GaussianMixture::setNumComponents(int n_components) {
    n_components_ = n_components;
}

void GaussianMixture::setCovarianceType(const std::string& type) {
    covariance_type_ = type;
}

FitResult GaussianMixture::fitEM(const Eigen::MatrixXd& data, int max_iterations) {
    int n_samples = data.rows();
    int n_features = data.cols();
    
    // Initialize parameters
    initializeParameters(data);
    
    FitResult result;
    double prev_log_likelihood = -std::numeric_limits<double>::infinity();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // E-step: compute responsibilities
        Eigen::MatrixXd responsibilities = computeResponsibilities(data);
        
        // M-step: update parameters
        updateParameters(data, responsibilities);
        
        // Compute log likelihood
        double log_likelihood = computeLogLikelihood(data);
        
        // Check convergence
        if (std::abs(log_likelihood - prev_log_likelihood) < 1e-4) {
            result.converged = true;
            break;
        }
        
        prev_log_likelihood = log_likelihood;
        result.iterations = iter + 1;
    }
    
    // Fill result
    result.n_components = n_components_;
    result.log_likelihood = prev_log_likelihood;
    result.bic = computeBIC(data, prev_log_likelihood);
    result.aic = computeAIC(data, prev_log_likelihood);
    result.weights = weights_;
    result.means = means_;
    result.covariances = covariances_;
    
    return result;
}

FitResult GaussianMixture::fitVariational(const Eigen::MatrixXd& data, int max_iterations) {
    // Simplified variational inference implementation
    // In practice, this would be more sophisticated
    return fitEM(data, max_iterations);
}

std::vector<int> GaussianMixture::predict(const Eigen::MatrixXd& data) const {
    Eigen::MatrixXd responsibilities = computeResponsibilities(data);
    std::vector<int> labels(data.rows());
    
    for (int i = 0; i < data.rows(); ++i) {
        responsibilities.row(i).maxCoeff(&labels[i]);
    }
    
    return labels;
}

Eigen::MatrixXd GaussianMixture::predictProba(const Eigen::MatrixXd& data) const {
    return computeResponsibilities(data);
}

std::vector<double> GaussianMixture::computeDensity(const Eigen::MatrixXd& data) const {
    std::vector<double> densities(data.rows());
    
    for (int i = 0; i < data.rows(); ++i) {
        double density = 0.0;
        for (int k = 0; k < n_components_; ++k) {
            density += weights_[k] * computeGaussianDensity(data.row(i), means_[k], covariances_[k]);
        }
        densities[i] = density;
    }
    
    return densities;
}

SampleResult GaussianMixture::sample(int n_samples, bool return_components) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    SampleResult result;
    result.samples = Eigen::MatrixXd(n_samples, means_[0].size());
    
    // Sample component assignments
    std::discrete_distribution<> component_dist(weights_.begin(), weights_.end());
    
    if (return_components) {
        result.component_labels.resize(n_samples);
    }
    
    for (int i = 0; i < n_samples; ++i) {
        int component = component_dist(gen);
        
        if (return_components) {
            result.component_labels[i] = component;
        }
        
        // Sample from the selected component
        result.samples.row(i) = sampleFromGaussian(means_[component], covariances_[component], gen);
    }
    
    return result;
}

void GaussianMixture::setParameters(const nlohmann::json& params) {
    if (params.contains("weights")) {
        weights_ = params["weights"].get<std::vector<double>>();
        n_components_ = weights_.size();
    }
    
    if (params.contains("means")) {
        means_.clear();
        for (const auto& mean : params["means"]) {
            means_.push_back(mean.get<Eigen::VectorXd>());
        }
    }
    
    if (params.contains("covariances")) {
        covariances_.clear();
        for (const auto& cov : params["covariances"]) {
            covariances_.push_back(cov.get<Eigen::MatrixXd>());
        }
    }
}

void GaussianMixture::initializeParameters(const Eigen::MatrixXd& data) {
    int n_samples = data.rows();
    int n_features = data.cols();
    
    // Initialize using k-means++
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Initialize weights uniformly
    weights_.resize(n_components_);
    std::fill(weights_.begin(), weights_.end(), 1.0 / n_components_);
    
    // Initialize means using k-means++ strategy
    means_.resize(n_components_);
    std::vector<int> center_indices;
    
    // Choose first center randomly
    std::uniform_int_distribution<> dis(0, n_samples - 1);
    center_indices.push_back(dis(gen));
    means_[0] = data.row(center_indices[0]).transpose();
    
    // Choose remaining centers
    for (int k = 1; k < n_components_; ++k) {
        std::vector<double> distances(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < k; ++j) {
                double dist = (data.row(i).transpose() - means_[j]).squaredNorm();
                min_dist = std::min(min_dist, dist);
            }
            distances[i] = min_dist;
        }
        
        std::discrete_distribution<> dist_distribution(distances.begin(), distances.end());
        int next_center = dist_distribution(gen);
        center_indices.push_back(next_center);
        means_[k] = data.row(next_center).transpose();
    }
    
    // Initialize covariances
    covariances_.resize(n_components_);
    Eigen::MatrixXd global_cov = computeCovariance(data);
    
    for (int k = 0; k < n_components_; ++k) {
        if (covariance_type_ == "full") {
            covariances_[k] = global_cov;
        } else if (covariance_type_ == "diag") {
            covariances_[k] = global_cov.diagonal().asDiagonal();
        } else if (covariance_type_ == "spherical") {
            double variance = global_cov.diagonal().mean();
            covariances_[k] = Eigen::MatrixXd::Identity(n_features, n_features) * variance;
        }
        
        // Add regularization
        covariances_[k] += Eigen::MatrixXd::Identity(n_features, n_features) * reg_covar_;
    }
}

Eigen::MatrixXd GaussianMixture::computeResponsibilities(const Eigen::MatrixXd& data) const {
    int n_samples = data.rows();
    Eigen::MatrixXd responsibilities(n_samples, n_components_);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int k = 0; k < n_components_; ++k) {
            responsibilities(i, k) = weights_[k] * 
                computeGaussianDensity(data.row(i), means_[k], covariances_[k]);
        }
        
        // Normalize
        double row_sum = responsibilities.row(i).sum();
        if (row_sum > 0) {
            responsibilities.row(i) /= row_sum;
        } else {
            responsibilities.row(i).setConstant(1.0 / n_components_);
        }
    }
    
    return responsibilities;
}

void GaussianMixture::updateParameters(const Eigen::MatrixXd& data, 
                                      const Eigen::MatrixXd& responsibilities) {
    int n_samples = data.rows();
    int n_features = data.cols();
    
    for (int k = 0; k < n_components_; ++k) {
        double Nk = responsibilities.col(k).sum();
        
        // Update weight
        weights_[k] = Nk / n_samples;
        
        // Update mean
        means_[k] = (responsibilities.col(k).transpose() * data).transpose() / Nk;
        
        // Update covariance
        Eigen::MatrixXd centered(n_samples, n_features);
        for (int i = 0; i < n_samples; ++i) {
            centered.row(i) = data.row(i) - means_[k].transpose();
        }
        
        if (covariance_type_ == "full") {
            covariances_[k] = Eigen::MatrixXd::Zero(n_features, n_features);
            for (int i = 0; i < n_samples; ++i) {
                covariances_[k] += responsibilities(i, k) * 
                    centered.row(i).transpose() * centered.row(i);
            }
            covariances_[k] /= Nk;
        } else if (covariance_type_ == "diag") {
            Eigen::VectorXd diag = Eigen::VectorXd::Zero(n_features);
            for (int i = 0; i < n_samples; ++i) {
                diag += responsibilities(i, k) * centered.row(i).array().square().matrix();
            }
            covariances_[k] = (diag / Nk).asDiagonal();
        } else if (covariance_type_ == "spherical") {
            double variance = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                variance += responsibilities(i, k) * centered.row(i).squaredNorm();
            }
            variance /= (Nk * n_features);
            covariances_[k] = Eigen::MatrixXd::Identity(n_features, n_features) * variance;
        }
        
        // Add regularization
        covariances_[k] += Eigen::MatrixXd::Identity(n_features, n_features) * reg_covar_;
    }
}

double GaussianMixture::computeLogLikelihood(const Eigen::MatrixXd& data) const {
    double log_likelihood = 0.0;
    
    for (int i = 0; i < data.rows(); ++i) {
        double sample_likelihood = 0.0;
        for (int k = 0; k < n_components_; ++k) {
            sample_likelihood += weights_[k] * 
                computeGaussianDensity(data.row(i), means_[k], covariances_[k]);
        }
        log_likelihood += std::log(sample_likelihood);
    }
    
    return log_likelihood;
}

double GaussianMixture::computeGaussianDensity(const Eigen::VectorXd& x,
                                               const Eigen::VectorXd& mean,
                                               const Eigen::MatrixXd& cov) const {
    int d = x.size();
    Eigen::VectorXd diff = x - mean;
    
    double det = cov.determinant();
    if (det <= 0) {
        return 1e-10;  // Small value to avoid numerical issues
    }
    
    Eigen::MatrixXd inv_cov = cov.inverse();
    double exponent = -0.5 * diff.transpose() * inv_cov * diff;
    double normalization = std::pow(2 * M_PI, -d/2.0) * std::pow(det, -0.5);
    
    return normalization * std::exp(exponent);
}

Eigen::MatrixXd GaussianMixture::computeCovariance(const Eigen::MatrixXd& data) const {
    Eigen::VectorXd mean = data.colwise().mean();
    Eigen::MatrixXd centered = data.rowwise() - mean.transpose();
    return (centered.transpose() * centered) / (data.rows() - 1);
}

double GaussianMixture::computeBIC(const Eigen::MatrixXd& data, double log_likelihood) const {
    int n_samples = data.rows();
    int n_features = data.cols();
    
    // Count parameters
    int n_params = n_components_ - 1;  // weights (minus 1 for constraint)
    n_params += n_components_ * n_features;  // means
    
    if (covariance_type_ == "full") {
        n_params += n_components_ * n_features * (n_features + 1) / 2;
    } else if (covariance_type_ == "diag") {
        n_params += n_components_ * n_features;
    } else if (covariance_type_ == "spherical") {
        n_params += n_components_;
    }
    
    return -2 * log_likelihood + n_params * std::log(n_samples);
}

double GaussianMixture::computeAIC(const Eigen::MatrixXd& data, double log_likelihood) const {
    int n_features = data.cols();
    
    // Count parameters (same as BIC)
    int n_params = n_components_ - 1;
    n_params += n_components_ * n_features;
    
    if (covariance_type_ == "full") {
        n_params += n_components_ * n_features * (n_features + 1) / 2;
    } else if (covariance_type_ == "diag") {
        n_params += n_components_ * n_features;
    } else if (covariance_type_ == "spherical") {
        n_params += n_components_;
    }
    
    return -2 * log_likelihood + 2 * n_params;
}

Eigen::VectorXd GaussianMixture::sampleFromGaussian(const Eigen::VectorXd& mean,
                                                    const Eigen::MatrixXd& cov,
                                                    std::mt19937& gen) const {
    std::normal_distribution<> dist(0.0, 1.0);
    Eigen::VectorXd z(mean.size());
    
    for (int i = 0; i < mean.size(); ++i) {
        z(i) = dist(gen);
    }
    
    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    return mean + llt.matrixL() * z;
}