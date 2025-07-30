#ifndef HYPERGEOMETRIC_FUNCTIONS_HPP
#define HYPERGEOMETRIC_FUNCTIONS_HPP

class HypergeometricFunctions {
public:
    // Confluent hypergeometric function 1F1(a; b; x)
    static double hypergeometric1F1(double a, double b, double x);
    
    // Gauss hypergeometric function 2F1(a, b; c; x)
    static double hypergeometric2F1(double a, double b, double c, double x);
    
    // Generalized hypergeometric function pFq
    static double hypergeometricPFQ(const double* a, int p, const double* b, int q, double x);
    
private:
    // Series expansion for 1F1
    static double hypergeometric1F1Series(double a, double b, double x);
    
    // Series expansion for 2F1
    static double hypergeometric2F1Series(double a, double b, double c, double x);
    
    // Kummer transformation for 1F1
    static double kummerTransform(double a, double b, double x);
    
    // Check convergence radius
    static bool checkConvergence(int p, int q, double x);
};