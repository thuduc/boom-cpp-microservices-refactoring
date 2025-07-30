#ifndef CLUSTERING_UTILS_HPP
#define CLUSTERING_UTILS_HPP

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>

class ClusteringUtils {
public:
    // JSON conversion utilities
    static Eigen::MatrixXd jsonToMatrix(const nlohmann::json& json_data);
    static nlohmann::json matrixToJson(const Eigen::MatrixXd& matrix);
    static nlohmann::json vectorsToJson(const std::vector<Eigen::VectorXd>& vectors);
    static nlohmann::json matricesToJson(const std::vector<Eigen::MatrixXd>& matrices);
    
    // Clustering metrics
    static double silhouetteScore(const Eigen::MatrixXd& data, const std::vector<int>& labels);
    static double daviesBouldinIndex(const Eigen::MatrixXd& data, const std::vector<int>& labels);
    static double calinskiHarabaszScore(const Eigen::MatrixXd& data, const std::vector<int>& labels);
    
    // Supervised metrics
    static double adjustedRandIndex(const std::vector<int>& labels1, const std::vector<int>& labels2);
    static double normalizedMutualInfo(const std::vector<int>& labels1, const std::vector<int>& labels2);
    static double homogeneityScore(const std::vector<int>& labels1, const std::vector<int>& labels2);
    static double completenessScore(const std::vector<int>& labels1, const std::vector<int>& labels2);
    
private:
    static double computeEntropy(const std::vector<int>& labels);
    static double computeMutualInfo(const std::vector<int>& labels1, const std::vector<int>& labels2);
};