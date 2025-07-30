#ifndef GAMMA_FUNCTIONS_HPP
#define GAMMA_FUNCTIONS_HPP

class GammaFunctions {
public:
    // Gamma function
    static double gamma(double x);
    
    // Natural logarithm of gamma function
    static double logGamma(double x);
    
    // Digamma function (psi function)
    static double digamma(double x);
    
    // Trigamma function
    static double trigamma(double x);
    
    // Incomplete gamma functions
    static double lowerIncompleteGamma(double a, double x);
    static double upperIncompleteGamma(double a, double x);
    
    // Regularized gamma functions
    static double regularizedGammaP(double a, double x);
    static double regularizedGammaQ(double a, double x);
    
private:
    // Helper functions
    static double lanczosApproximation(double z);
    static double stirlingApproximation(double x);
};