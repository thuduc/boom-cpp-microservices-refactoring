#ifndef SOLVERS_HPP
#define SOLVERS_HPP

#include <Eigen/Dense>

class Solvers {
public:
    // General linear system solver (Ax = b)
    static Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    
    // Solve using Cholesky decomposition (for symmetric positive definite matrices)
    static Eigen::VectorXd solveCholesky(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    
    // Solve using LU decomposition
    static Eigen::VectorXd solveLU(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    
    // Solve using QR decomposition
    static Eigen::VectorXd solveQR(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
    
    // Least squares solver (for overdetermined systems)
    static Eigen::VectorXd leastSquares(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
};

#endif // SOLVERS_HPP