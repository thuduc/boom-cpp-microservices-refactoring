#include "core/matrix_operations.hpp"
#include <stdexcept>

Eigen::MatrixXd MatrixOperations::multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    return A * B;
}

Eigen::MatrixXd MatrixOperations::add(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    return A + B;
}

Eigen::MatrixXd MatrixOperations::subtract(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    return A - B;
}

Eigen::MatrixXd MatrixOperations::transpose(const Eigen::MatrixXd& A) {
    return A.transpose();
}

Eigen::MatrixXd MatrixOperations::inverse(const Eigen::MatrixXd& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for inversion");
    }
    
    if (std::abs(A.determinant()) < 1e-10) {
        throw std::runtime_error("Matrix is singular or nearly singular");
    }
    
    return A.inverse();
}

double MatrixOperations::determinant(const Eigen::MatrixXd& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for determinant");
    }
    return A.determinant();
}

double MatrixOperations::trace(const Eigen::MatrixXd& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for trace");
    }
    return A.trace();
}

double MatrixOperations::norm(const Eigen::MatrixXd& A, const std::string& type) {
    if (type == "frobenius" || type == "fro") {
        return A.norm();
    } else if (type == "1") {
        return A.lpNorm<1>();
    } else if (type == "inf") {
        return A.lpNorm<Eigen::Infinity>();
    } else {
        throw std::invalid_argument("Unknown norm type: " + type);
    }
}