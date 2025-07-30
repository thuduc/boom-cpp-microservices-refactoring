#ifndef DECOMPOSITIONS_HPP
#define DECOMPOSITIONS_HPP

#include <Eigen/Dense>
#include <tuple>
#include <vector>

class Decompositions {
public:
    // Cholesky decomposition (A = L * L^T)
    static Eigen::MatrixXd cholesky(const Eigen::MatrixXd& A);
    
    // LU decomposition (P * A = L * U)
    static std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> 
        lu(const Eigen::MatrixXd& A);
    
    // QR decomposition (A = Q * R)
    static std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
        qr(const Eigen::MatrixXd& A);
    
    // SVD decomposition (A = U * S * V^T)
    static std::tuple<Eigen::MatrixXd, std::vector<double>, Eigen::MatrixXd> 
        svd(const Eigen::MatrixXd& A);
    
    // Eigenvalue decomposition
    static std::tuple<Eigen::VectorXcd, Eigen::MatrixXcd> 
        eigen(const Eigen::MatrixXd& A, bool compute_eigenvectors = true);
    
    // Helper to compute rank from singular values
    static int rank(const std::vector<double>& singular_values, double tolerance = 1e-10);
};

#endif // DECOMPOSITIONS_HPP