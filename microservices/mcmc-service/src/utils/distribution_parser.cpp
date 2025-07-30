#include "distribution_parser.hpp"
#include <cmath>
#include <Eigen/Dense>

TargetDistribution DistributionParser::createNormalDistribution(
    double mean, double std_dev) {
    
    double log_norm_const = -0.5 * std::log(2 * M_PI) - std::log(std_dev);
    double inv_var = 1.0 / (std_dev * std_dev);
    
    return [mean, log_norm_const, inv_var](const std::vector<double>& x) {
        if (x.size() != 1) {
            throw std::invalid_argument("Univariate normal expects 1D input");
        }
        double z = x[0] - mean;
        return log_norm_const - 0.5 * inv_var * z * z;
    };
}

TargetDistribution DistributionParser::createMultivariateNormal(
    const std::vector<double>& mean, 
    const std::vector<std::vector<double>>& covariance) {
    
    size_t dim = mean.size();
    
    // Convert to Eigen
    Eigen::VectorXd mu(dim);
    for (size_t i = 0; i < dim; ++i) {
        mu(i) = mean[i];
    }
    
    Eigen::MatrixXd sigma(dim, dim);
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            sigma(i, j) = covariance[i][j];
        }
    }
    
    // Compute precision matrix and log normalizing constant
    Eigen::LLT<Eigen::MatrixXd> llt(sigma);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument("Covariance matrix is not positive definite");
    }
    
    Eigen::MatrixXd precision = sigma.inverse();
    double log_det = std::log(sigma.determinant());
    double log_norm_const = -0.5 * (dim * std::log(2 * M_PI) + log_det);
    
    return [mu, precision, log_norm_const, dim](const std::vector<double>& x) {
        if (x.size() != dim) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        Eigen::VectorXd diff(dim);
        for (size_t i = 0; i < dim; ++i) {
            diff(i) = x[i] - mu(i);
        }
        
        double quad_form = diff.transpose() * precision * diff;
        return log_norm_const - 0.5 * quad_form;
    };
}

TargetDistribution DistributionParser::createMixtureDistribution(
    const std::vector<double>& weights,
    const std::vector<TargetDistribution>& components) {
    
    // Normalize weights and convert to log
    std::vector<double> log_weights(weights.size());
    double sum_weights = 0.0;
    for (double w : weights) {
        sum_weights += w;
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        log_weights[i] = std::log(weights[i] / sum_weights);
    }
    
    return [log_weights, components](const std::vector<double>& x) {
        // Log-sum-exp trick for numerical stability
        std::vector<double> log_probs(components.size());
        double max_log_prob = -std::numeric_limits<double>::infinity();
        
        for (size_t i = 0; i < components.size(); ++i) {
            log_probs[i] = log_weights[i] + components[i](x);
            max_log_prob = std::max(max_log_prob, log_probs[i]);
        }
        
        double sum_exp = 0.0;
        for (double log_p : log_probs) {
            sum_exp += std::exp(log_p - max_log_prob);
        }
        
        return max_log_prob + std::log(sum_exp);
    };
}

TargetDistribution DistributionParser::parseDistribution(const json& dist_json) {
    if (dist_json.is_string()) {
        std::string name = dist_json.get<std::string>();
        
        // Standard distributions
        if (name == "standard_normal") {
            return createNormalDistribution(0.0, 1.0);
        } else if (name == "standard_cauchy") {
            return [](const std::vector<double>& x) {
                if (x.size() != 1) {
                    throw std::invalid_argument("Univariate Cauchy expects 1D input");
                }
                return -std::log(M_PI) - std::log(1 + x[0] * x[0]);
            };
        }
    } else if (dist_json.is_object()) {
        std::string type = dist_json["type"];
        
        if (type == "normal") {
            double mean = dist_json.value("mean", 0.0);
            double std_dev = dist_json.value("std_dev", 1.0);
            return createNormalDistribution(mean, std_dev);
            
        } else if (type == "multivariate_normal") {
            std::vector<double> mean = dist_json["mean"];
            std::vector<std::vector<double>> cov = dist_json["covariance"];
            return createMultivariateNormal(mean, cov);
            
        } else if (type == "mixture") {
            std::vector<double> weights = dist_json["weights"];
            std::vector<TargetDistribution> components;
            
            for (const auto& comp : dist_json["components"]) {
                components.push_back(parseDistribution(comp));
            }
            
            return createMixtureDistribution(weights, components);
            
        } else if (type == "custom") {
            // Custom density function (simplified)
            std::string expr = dist_json["expression"];
            
            if (expr == "banana") {
                // Banana-shaped distribution
                double b = dist_json.value("b", 0.1);
                return [b](const std::vector<double>& x) {
                    if (x.size() < 2) {
                        throw std::invalid_argument("Banana distribution needs at least 2D");
                    }
                    double term1 = x[0] * x[0];
                    double term2 = (x[1] - b * x[0] * x[0]) * (x[1] - b * x[0] * x[0]);
                    return -0.5 * (term1 + term2);
                };
            }
        }
    }
    
    throw std::invalid_argument("Unknown distribution format");
}