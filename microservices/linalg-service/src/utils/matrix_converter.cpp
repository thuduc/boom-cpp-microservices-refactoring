#include "utils/matrix_converter.hpp"
#include <stdexcept>

Eigen::MatrixXd MatrixConverter::fromJson(const json& j) {
    if (!j.is_array()) {
        throw std::invalid_argument("Matrix must be represented as array of arrays");
    }
    
    if (j.empty()) {
        return Eigen::MatrixXd(0, 0);
    }
    
    size_t rows = j.size();
    size_t cols = 0;
    
    // Determine number of columns from first row
    if (j[0].is_array()) {
        cols = j[0].size();
    } else {
        throw std::invalid_argument("Matrix rows must be arrays");
    }
    
    Eigen::MatrixXd matrix(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        if (!j[i].is_array() || j[i].size() != cols) {
            throw std::invalid_argument("All matrix rows must have the same number of columns");
        }
        
        for (size_t j_idx = 0; j_idx < cols; ++j_idx) {
            matrix(i, j_idx) = j[i][j_idx].get<double>();
        }
    }
    
    return matrix;
}

json MatrixConverter::toJson(const Eigen::MatrixXd& matrix) {
    json result = json::array();
    
    for (int i = 0; i < matrix.rows(); ++i) {
        json row = json::array();
        for (int j = 0; j < matrix.cols(); ++j) {
            row.push_back(matrix(i, j));
        }
        result.push_back(row);
    }
    
    return result;
}

Eigen::VectorXd MatrixConverter::vectorFromJson(const json& j) {
    if (!j.is_array()) {
        throw std::invalid_argument("Vector must be represented as array");
    }
    
    Eigen::VectorXd vector(j.size());
    
    for (size_t i = 0; i < j.size(); ++i) {
        vector(i) = j[i].get<double>();
    }
    
    return vector;
}

json MatrixConverter::vectorToJson(const Eigen::VectorXd& vector) {
    json result = json::array();
    
    for (int i = 0; i < vector.size(); ++i) {
        result.push_back(vector(i));
    }
    
    return result;
}

json MatrixConverter::complexVectorToJson(const Eigen::VectorXcd& vector) {
    json result = json::array();
    
    for (int i = 0; i < vector.size(); ++i) {
        json complex_num = {
            {"real", vector(i).real()},
            {"imag", vector(i).imag()}
        };
        result.push_back(complex_num);
    }
    
    return result;
}

json MatrixConverter::complexMatrixToJson(const Eigen::MatrixXcd& matrix) {
    json result = json::array();
    
    for (int i = 0; i < matrix.rows(); ++i) {
        json row = json::array();
        for (int j = 0; j < matrix.cols(); ++j) {
            json complex_num = {
                {"real", matrix(i, j).real()},
                {"imag", matrix(i, j).imag()}
            };
            row.push_back(complex_num);
        }
        result.push_back(row);
    }
    
    return result;
}