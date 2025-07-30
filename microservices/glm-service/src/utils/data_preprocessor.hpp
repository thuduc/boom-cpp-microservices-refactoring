#ifndef DATA_PREPROCESSOR_HPP
#define DATA_PREPROCESSOR_HPP

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

class DataPreprocessor {
public:
    // JSON to Eigen conversions
    static Eigen::MatrixXd jsonToMatrix(const json& data);
    static Eigen::VectorXd jsonToVector(const json& data);
    
    // Eigen to JSON conversions
    static json matrixToJson(const Eigen::MatrixXd& matrix);
    static json vectorToJson(const Eigen::VectorXd& vector);
    
    // Data validation
    static void validateMatrix(const Eigen::MatrixXd& X);
    static void validateVector(const Eigen::VectorXd& y);
    static void validateBinaryOutcome(const Eigen::VectorXd& y);
    static void validateCountOutcome(const Eigen::VectorXd& y);
    
    // Data preprocessing
    static Eigen::MatrixXd addPolynomialFeatures(const Eigen::MatrixXd& X, int degree);
    static Eigen::MatrixXd addInteractionTerms(const Eigen::MatrixXd& X);
    static std::pair<Eigen::MatrixXd, Eigen::VectorXd> removeNaNs(
        const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

#endif // DATA_PREPROCESSOR_HPP