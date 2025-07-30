#ifndef BETA_FUNCTIONS_HPP
#define BETA_FUNCTIONS_HPP

class BetaFunctions {
public:
    // Beta function B(a, b)
    static double beta(double a, double b);
    
    // Natural logarithm of beta function
    static double logBeta(double a, double b);
    
    // Regularized incomplete beta function I_x(a, b)
    static double regularizedBeta(double x, double a, double b);
    
    // Inverse of regularized beta function
    static double inverseBeta(double p, double a, double b);
    
private:
    // Helper functions
    static double betaContinuedFraction(double x, double a, double b);
};