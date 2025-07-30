#ifndef MATRIX_CONVERTER_HPP
#define MATRIX_CONVERTER_HPP

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class MatrixConverter {
public:
    // Convert JSON to Eigen matrix
    static Eigen::MatrixXd fromJson(const json& j);
    
    // Convert Eigen matrix to JSON
    static json toJson(const Eigen::MatrixXd& matrix);
    
    // Convert JSON to Eigen vector
    static Eigen::VectorXd vectorFromJson(const json& j);
    
    // Convert Eigen vector to JSON
    static json vectorToJson(const Eigen::VectorXd& vector);
    
    // Convert complex vector to JSON
    static json complexVectorToJson(const Eigen::VectorXcd& vector);
    
    // Convert complex matrix to JSON
    static json complexMatrixToJson(const Eigen::MatrixXcd& matrix);
};

#endif // MATRIX_CONVERTER_HPP