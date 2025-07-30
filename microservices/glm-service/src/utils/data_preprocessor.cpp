#include "data_preprocessor.hpp"
#include <stdexcept>
#include <cmath>

Eigen::MatrixXd DataPreprocessor::jsonToMatrix(const json& data) {
    if (!data.is_array()) {
        throw std::runtime_error("Expected array for matrix data");
    }
    
    if (data.empty()) {
        throw std::runtime_error("Empty matrix data");
    }
    
    // Check if it's a 1D or 2D array
    if (data[0].is_number()) {
        // 1D array - convert to column vector
        Eigen::MatrixXd matrix(data.size(), 1);
        for (size_t i = 0; i < data.size(); ++i) {
            matrix(i, 0) = data[i].get<double>();
        }
        return matrix;
    } else if (data[0].is_array()) {
        // 2D array
        size_t rows = data.size();
        size_t cols = data[0].size();
        
        Eigen::MatrixXd matrix(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            if (!data[i].is_array() || data[i].size() != cols) {
                throw std::runtime_error("Inconsistent matrix dimensions");
            }
            for (size_t j = 0; j < cols; ++j) {
                matrix(i, j) = data[i][j].get<double>();
            }
        }
        return matrix;
    } else {
        throw std::runtime_error("Invalid matrix format");
    }
}

Eigen::VectorXd DataPreprocessor::jsonToVector(const json& data) {
    if (!data.is_array()) {
        throw std::runtime_error("Expected array for vector data");
    }
    
    Eigen::VectorXd vector(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (!data[i].is_number()) {
            throw std::runtime_error("Non-numeric value in vector");
        }
        vector(i) = data[i].get<double>();
    }
    
    return vector;
}

json DataPreprocessor::matrixToJson(const Eigen::MatrixXd& matrix) {
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

json DataPreprocessor::vectorToJson(const Eigen::VectorXd& vector) {
    json result = json::array();
    
    for (int i = 0; i < vector.size(); ++i) {
        result.push_back(vector(i));
    }
    
    return result;
}

void DataPreprocessor::validateMatrix(const Eigen::MatrixXd& X) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::runtime_error("Empty feature matrix");
    }
    
    // Check for NaN or infinite values
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            if (std::isnan(X(i, j)) || std::isinf(X(i, j))) {
                throw std::runtime_error("Feature matrix contains NaN or infinite values");
            }
        }
    }
}

void DataPreprocessor::validateVector(const Eigen::VectorXd& y) {
    if (y.size() == 0) {
        throw std::runtime_error("Empty response vector");
    }
    
    // Check for NaN or infinite values
    for (int i = 0; i < y.size(); ++i) {
        if (std::isnan(y(i)) || std::isinf(y(i))) {
            throw std::runtime_error("Response vector contains NaN or infinite values");
        }
    }
}

void DataPreprocessor::validateBinaryOutcome(const Eigen::VectorXd& y) {
    validateVector(y);
    
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) != 0.0 && y(i) != 1.0) {
            throw std::runtime_error("Binary outcome must contain only 0 and 1 values");
        }
    }
}

void DataPreprocessor::validateCountOutcome(const Eigen::VectorXd& y) {
    validateVector(y);
    
    for (int i = 0; i < y.size(); ++i) {
        if (y(i) < 0 || std::floor(y(i)) != y(i)) {
            throw std::runtime_error("Count outcome must contain non-negative integers");
        }
    }
}

Eigen::MatrixXd DataPreprocessor::addPolynomialFeatures(const Eigen::MatrixXd& X, int degree) {
    if (degree < 1) {
        throw std::runtime_error("Polynomial degree must be at least 1");
    }
    
    int n = X.rows();
    int p = X.cols();
    
    // Calculate total number of polynomial features
    int total_features = 0;
    for (int d = 1; d <= degree; ++d) {
        total_features += p;  // Simplified - actual calculation would be more complex
    }
    
    Eigen::MatrixXd X_poly(n, total_features);
    
    // Copy original features
    X_poly.leftCols(p) = X;
    
    // Add polynomial terms
    int col_idx = p;
    for (int d = 2; d <= degree; ++d) {
        for (int j = 0; j < p; ++j) {
            X_poly.col(col_idx) = X.col(j).array().pow(d);
            col_idx++;
        }
    }
    
    return X_poly;
}

Eigen::MatrixXd DataPreprocessor::addInteractionTerms(const Eigen::MatrixXd& X) {
    int n = X.rows();
    int p = X.cols();
    
    // Number of interaction terms: p choose 2
    int n_interactions = p * (p - 1) / 2;
    
    Eigen::MatrixXd X_with_interactions(n, p + n_interactions);
    
    // Copy original features
    X_with_interactions.leftCols(p) = X;
    
    // Add interaction terms
    int col_idx = p;
    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j) {
            X_with_interactions.col(col_idx) = X.col(i).array() * X.col(j).array();
            col_idx++;
        }
    }
    
    return X_with_interactions;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> DataPreprocessor::removeNaNs(
    const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    
    std::vector<int> valid_indices;
    
    for (int i = 0; i < X.rows(); ++i) {
        bool has_nan = false;
        
        // Check X
        for (int j = 0; j < X.cols(); ++j) {
            if (std::isnan(X(i, j)) || std::isinf(X(i, j))) {
                has_nan = true;
                break;
            }
        }
        
        // Check y
        if (!has_nan && (std::isnan(y(i)) || std::isinf(y(i)))) {
            has_nan = true;
        }
        
        if (!has_nan) {
            valid_indices.push_back(i);
        }
    }
    
    if (valid_indices.empty()) {
        throw std::runtime_error("No valid observations after removing NaNs");
    }
    
    // Create cleaned matrices
    Eigen::MatrixXd X_clean(valid_indices.size(), X.cols());
    Eigen::VectorXd y_clean(valid_indices.size());
    
    for (size_t i = 0; i < valid_indices.size(); ++i) {
        X_clean.row(i) = X.row(valid_indices[i]);
        y_clean(i) = y(valid_indices[i]);
    }
    
    return {X_clean, y_clean};
}