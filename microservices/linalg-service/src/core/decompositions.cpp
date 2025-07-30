#include "core/decompositions.hpp"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <stdexcept>

Eigen::MatrixXd Decompositions::cholesky(const Eigen::MatrixXd& A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for Cholesky decomposition");
    }
    
    // Check if matrix is symmetric
    if (!A.isApprox(A.transpose())) {
        throw std::invalid_argument("Matrix must be symmetric for Cholesky decomposition");
    }
    
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky decomposition failed - matrix may not be positive definite");
    }
    
    return llt.matrixL();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> 
Decompositions::lu(const Eigen::MatrixXd& A) {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    
    // Extract L and U
    Eigen::MatrixXd LU = lu.matrixLU();
    Eigen::MatrixXd L = Eigen::MatrixXd::Identity(A.rows(), A.rows());
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    
    // Fill L and U
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (i > j) {
                L(i, j) = LU(i, j);
            } else {
                U(i, j) = LU(i, j);
            }
        }
    }
    
    // Get permutation matrix
    Eigen::MatrixXd P = lu.permutationP();
    
    return std::make_tuple(L, U, P);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
Decompositions::qr(const Eigen::MatrixXd& A) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    
    // Resize R to match original dimensions
    if (A.rows() > A.cols()) {
        R.conservativeResize(A.cols(), A.cols());
    }
    
    return std::make_tuple(Q, R);
}

std::tuple<Eigen::MatrixXd, std::vector<double>, Eigen::MatrixXd> 
Decompositions::svd(const Eigen::MatrixXd& A) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::VectorXd s = svd.singularValues();
    
    // Convert singular values to std::vector
    std::vector<double> singular_values(s.size());
    for (int i = 0; i < s.size(); ++i) {
        singular_values[i] = s(i);
    }
    
    return std::make_tuple(U, singular_values, V);
}

std::tuple<Eigen::VectorXcd, Eigen::MatrixXcd> 
Decompositions::eigen(const Eigen::MatrixXd& A, bool compute_eigenvectors) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue decomposition");
    }
    
    if (compute_eigenvectors) {
        Eigen::EigenSolver<Eigen::MatrixXd> solver(A);
        return std::make_tuple(solver.eigenvalues(), solver.eigenvectors());
    } else {
        Eigen::EigenSolver<Eigen::MatrixXd> solver(A, false);
        return std::make_tuple(solver.eigenvalues(), Eigen::MatrixXcd());
    }
}

int Decompositions::rank(const std::vector<double>& singular_values, double tolerance) {
    int rank = 0;
    for (double s : singular_values) {
        if (s > tolerance) {
            rank++;
        }
    }
    return rank;
}