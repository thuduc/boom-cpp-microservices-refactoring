#include "core/solvers.hpp"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/QR>
#include <stdexcept>

Eigen::VectorXd Solvers::solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != b.rows()) {
        throw std::invalid_argument("Matrix A and vector b dimensions incompatible");
    }
    
    if (A.rows() != A.cols()) {
        // Overdetermined or underdetermined system - use least squares
        return leastSquares(A, b);
    }
    
    // Square system - use LU decomposition
    return solveLU(A, b);
}

Eigen::VectorXd Solvers::solveCholesky(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for Cholesky solver");
    }
    
    if (A.rows() != b.rows()) {
        throw std::invalid_argument("Matrix A and vector b dimensions incompatible");
    }
    
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky decomposition failed - matrix may not be positive definite");
    }
    
    return llt.solve(b);
}

Eigen::VectorXd Solvers::solveLU(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square for LU solver");
    }
    
    if (A.rows() != b.rows()) {
        throw std::invalid_argument("Matrix A and vector b dimensions incompatible");
    }
    
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    
    if (std::abs(A.determinant()) < 1e-10) {
        throw std::runtime_error("Matrix is singular or nearly singular");
    }
    
    return lu.solve(b);
}

Eigen::VectorXd Solvers::solveQR(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != b.rows()) {
        throw std::invalid_argument("Matrix A and vector b dimensions incompatible");
    }
    
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    return qr.solve(b);
}

Eigen::VectorXd Solvers::leastSquares(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    if (A.rows() != b.rows()) {
        throw std::invalid_argument("Matrix A and vector b dimensions incompatible");
    }
    
    // Use SVD for robust least squares solution
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.solve(b);
}