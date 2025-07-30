#ifndef DISTRIBUTION_PARSER_HPP
#define DISTRIBUTION_PARSER_HPP

#include <functional>
#include <vector>
#include <nlohmann/json.hpp>
#include "../core/mcmc_engine.hpp"

using json = nlohmann::json;

class DistributionParser {
public:
    // Parse a target distribution from JSON
    static TargetDistribution parseDistribution(const json& dist_json);
    
private:
    // Helper functions for common distributions
    static TargetDistribution createNormalDistribution(double mean, double std_dev);
    static TargetDistribution createMultivariateNormal(
        const std::vector<double>& mean, 
        const std::vector<std::vector<double>>& covariance);
    static TargetDistribution createMixtureDistribution(
        const std::vector<double>& weights,
        const std::vector<TargetDistribution>& components);
};

#endif // DISTRIBUTION_PARSER_HPP