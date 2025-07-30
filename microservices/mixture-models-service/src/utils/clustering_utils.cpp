#include "clustering_utils.hpp"
#include <cmath>
#include <algorithm>
#include <map>
#include <set>

Eigen::MatrixXd ClusteringUtils::jsonToMatrix(const nlohmann::json& json_data) {
    if (!json_data.is_array() || json_data.empty()) {
        return Eigen::MatrixXd();
    }
    
    int rows = json_data.size();
    int cols = json_data[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = json_data[i][j].get<double>();
        }
    }
    
    return matrix;
}

nlohmann::json ClusteringUtils::matrixToJson(const Eigen::MatrixXd& matrix) {
    nlohmann::json result = nlohmann::json::array();
    
    for (int i = 0; i < matrix.rows(); ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j = 0; j < matrix.cols(); ++j) {
            row.push_back(matrix(i, j));
        }
        result.push_back(row);
    }
    
    return result;
}

nlohmann::json ClusteringUtils::vectorsToJson(const std::vector<Eigen::VectorXd>& vectors) {
    nlohmann::json result = nlohmann::json::array();
    
    for (const auto& vec : vectors) {
        nlohmann::json vec_json = nlohmann::json::array();
        for (int i = 0; i < vec.size(); ++i) {
            vec_json.push_back(vec(i));
        }
        result.push_back(vec_json);
    }
    
    return result;
}

nlohmann::json ClusteringUtils::matricesToJson(const std::vector<Eigen::MatrixXd>& matrices) {
    nlohmann::json result = nlohmann::json::array();
    
    for (const auto& mat : matrices) {
        result.push_back(matrixToJson(mat));
    }
    
    return result;
}

double ClusteringUtils::silhouetteScore(const Eigen::MatrixXd& data, const std::vector<int>& labels) {
    int n = data.rows();
    if (n < 2) return 0.0;
    
    std::vector<double> silhouettes(n);
    
    for (int i = 0; i < n; ++i) {
        int label_i = labels[i];
        
        // Compute a(i): mean distance to points in same cluster
        double a_i = 0.0;
        int same_cluster_count = 0;
        
        for (int j = 0; j < n; ++j) {
            if (i != j && labels[j] == label_i) {
                a_i += (data.row(i) - data.row(j)).norm();
                same_cluster_count++;
            }
        }
        
        if (same_cluster_count > 0) {
            a_i /= same_cluster_count;
        }
        
        // Compute b(i): min mean distance to points in other clusters
        std::map<int, double> other_cluster_dists;
        std::map<int, int> other_cluster_counts;
        
        for (int j = 0; j < n; ++j) {
            if (labels[j] != label_i) {
                other_cluster_dists[labels[j]] += (data.row(i) - data.row(j)).norm();
                other_cluster_counts[labels[j]]++;
            }
        }
        
        double b_i = std::numeric_limits<double>::max();
        for (const auto& [cluster, dist] : other_cluster_dists) {
            if (other_cluster_counts[cluster] > 0) {
                double mean_dist = dist / other_cluster_counts[cluster];
                b_i = std::min(b_i, mean_dist);
            }
        }
        
        // Compute silhouette coefficient
        if (std::max(a_i, b_i) > 0) {
            silhouettes[i] = (b_i - a_i) / std::max(a_i, b_i);
        } else {
            silhouettes[i] = 0.0;
        }
    }
    
    // Return mean silhouette score
    return std::accumulate(silhouettes.begin(), silhouettes.end(), 0.0) / n;
}

double ClusteringUtils::daviesBouldinIndex(const Eigen::MatrixXd& data, const std::vector<int>& labels) {
    // Compute cluster centers
    std::map<int, Eigen::VectorXd> centers;
    std::map<int, int> counts;
    
    for (int i = 0; i < data.rows(); ++i) {
        int label = labels[i];
        if (centers.find(label) == centers.end()) {
            centers[label] = Eigen::VectorXd::Zero(data.cols());
        }
        centers[label] += data.row(i).transpose();
        counts[label]++;
    }
    
    for (auto& [label, center] : centers) {
        center /= counts[label];
    }
    
    // Compute within-cluster scatter
    std::map<int, double> scatter;
    for (int i = 0; i < data.rows(); ++i) {
        int label = labels[i];
        scatter[label] += (data.row(i).transpose() - centers[label]).norm();
    }
    
    for (auto& [label, s] : scatter) {
        s /= counts[label];
    }
    
    // Compute Davies-Bouldin index
    double db_index = 0.0;
    int n_clusters = centers.size();
    
    for (const auto& [i, center_i] : centers) {
        double max_ratio = 0.0;
        
        for (const auto& [j, center_j] : centers) {
            if (i != j) {
                double dist = (center_i - center_j).norm();
                if (dist > 0) {
                    double ratio = (scatter[i] + scatter[j]) / dist;
                    max_ratio = std::max(max_ratio, ratio);
                }
            }
        }
        
        db_index += max_ratio;
    }
    
    return db_index / n_clusters;
}

double ClusteringUtils::calinskiHarabaszScore(const Eigen::MatrixXd& data, const std::vector<int>& labels) {
    int n = data.rows();
    int k = *std::max_element(labels.begin(), labels.end()) + 1;
    
    if (k == 1 || k == n) return 0.0;
    
    // Compute overall center
    Eigen::VectorXd overall_center = data.colwise().mean();
    
    // Compute cluster centers
    std::map<int, Eigen::VectorXd> centers;
    std::map<int, int> counts;
    
    for (int i = 0; i < n; ++i) {
        int label = labels[i];
        if (centers.find(label) == centers.end()) {
            centers[label] = Eigen::VectorXd::Zero(data.cols());
        }
        centers[label] += data.row(i).transpose();
        counts[label]++;
    }
    
    for (auto& [label, center] : centers) {
        center /= counts[label];
    }
    
    // Between-cluster sum of squares
    double between_ss = 0.0;
    for (const auto& [label, center] : centers) {
        between_ss += counts[label] * (center - overall_center).squaredNorm();
    }
    
    // Within-cluster sum of squares
    double within_ss = 0.0;
    for (int i = 0; i < n; ++i) {
        within_ss += (data.row(i).transpose() - centers[labels[i]]).squaredNorm();
    }
    
    return (between_ss / (k - 1)) / (within_ss / (n - k));
}

double ClusteringUtils::adjustedRandIndex(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    int n = labels1.size();
    if (n != labels2.size()) return 0.0;
    
    // Build contingency table
    std::map<std::pair<int, int>, int> contingency;
    std::map<int, int> sum1, sum2;
    
    for (int i = 0; i < n; ++i) {
        contingency[{labels1[i], labels2[i]}]++;
        sum1[labels1[i]]++;
        sum2[labels2[i]]++;
    }
    
    // Compute index
    double index = 0.0;
    for (const auto& [pair, count] : contingency) {
        if (count > 1) {
            index += count * (count - 1) / 2.0;
        }
    }
    
    double expected_index = 0.0;
    double max_index = 0.0;
    
    for (const auto& [label, count] : sum1) {
        if (count > 1) {
            max_index += count * (count - 1) / 2.0;
        }
    }
    
    for (const auto& [label, count] : sum2) {
        if (count > 1) {
            max_index += count * (count - 1) / 2.0;
        }
    }
    
    max_index /= 2.0;
    
    for (const auto& [label1, count1] : sum1) {
        for (const auto& [label2, count2] : sum2) {
            expected_index += count1 * count2;
        }
    }
    
    expected_index = expected_index * (expected_index - 1) / (2.0 * n * (n - 1));
    
    if (max_index == expected_index) return 0.0;
    
    return (index - expected_index) / (max_index - expected_index);
}

double ClusteringUtils::normalizedMutualInfo(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    double h1 = computeEntropy(labels1);
    double h2 = computeEntropy(labels2);
    double mi = computeMutualInfo(labels1, labels2);
    
    if (h1 == 0.0 && h2 == 0.0) return 1.0;
    if (h1 == 0.0 || h2 == 0.0) return 0.0;
    
    return 2.0 * mi / (h1 + h2);
}

double ClusteringUtils::homogeneityScore(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    double h2 = computeEntropy(labels2);
    if (h2 == 0.0) return 1.0;
    
    double mi = computeMutualInfo(labels1, labels2);
    return mi / h2;
}

double ClusteringUtils::completenessScore(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    double h1 = computeEntropy(labels1);
    if (h1 == 0.0) return 1.0;
    
    double mi = computeMutualInfo(labels1, labels2);
    return mi / h1;
}

double ClusteringUtils::computeEntropy(const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int label : labels) {
        counts[label]++;
    }
    
    double entropy = 0.0;
    int n = labels.size();
    
    for (const auto& [label, count] : counts) {
        if (count > 0) {
            double p = static_cast<double>(count) / n;
            entropy -= p * std::log2(p);
        }
    }
    
    return entropy;
}

double ClusteringUtils::computeMutualInfo(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    int n = labels1.size();
    
    std::map<std::pair<int, int>, int> joint_counts;
    std::map<int, int> counts1, counts2;
    
    for (int i = 0; i < n; ++i) {
        joint_counts[{labels1[i], labels2[i]}]++;
        counts1[labels1[i]]++;
        counts2[labels2[i]]++;
    }
    
    double mi = 0.0;
    
    for (const auto& [pair, joint_count] : joint_counts) {
        if (joint_count > 0) {
            double p_joint = static_cast<double>(joint_count) / n;
            double p1 = static_cast<double>(counts1[pair.first]) / n;
            double p2 = static_cast<double>(counts2[pair.second]) / n;
            
            mi += p_joint * std::log2(p_joint / (p1 * p2));
        }
    }
    
    return mi;
}