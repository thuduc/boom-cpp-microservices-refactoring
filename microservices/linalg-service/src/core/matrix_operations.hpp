#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <Eigen/Dense>

class MatrixOperations {
public:
    // Matrix multiplication
    static Eigen::MatrixXd multiply(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
    
    // Matrix addition
    static Eigen::MatrixXd add(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
    
    // Matrix subtraction
    static Eigen::MatrixXd subtract(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
    
    // Matrix transpose
    static Eigen::MatrixXd transpose(const Eigen::MatrixXd& A);
    
    // Matrix inverse
    static Eigen::MatrixXd inverse(const Eigen::MatrixXd& A);
    
    // Matrix determinant
    static double determinant(const Eigen::MatrixXd& A);
    
    // Matrix trace
    static double trace(const Eigen::MatrixXd& A);
    
    // Matrix norm
    static double norm(const Eigen::MatrixXd& A, const std::string& type = "frobenius");
};

#endif // MATRIX_OPERATIONS_HPP