#ifndef NUMERIC_UTILS_HPP
#define NUMERIC_UTILS_HPP

#include <vector>
#include <complex>

class NumericUtils {
public:
    // Numerical differentiation
    static double derivative(double (*f)(double), double x, double h = 1e-8);
    
    // Numerical integration
    static double integrate(double (*f)(double), double a, double b, int n = 1000);
    
    // Root finding
    static double findRoot(double (*f)(double), double a, double b, double tol = 1e-10);
    
    // Complex number utilities
    static std::complex<double> complexPower(const std::complex<double>& z, double n);
    static std::complex<double> complexLog(const std::complex<double>& z);
    
    // Precision utilities
    static bool almostEqual(double a, double b, double tol = 1e-10);
    static double roundToSignificant(double x, int digits);
};