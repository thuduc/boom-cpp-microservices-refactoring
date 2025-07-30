#ifndef POLYGAMMA_FUNCTIONS_HPP
#define POLYGAMMA_FUNCTIONS_HPP

class PolygammaFunctions {
public:
    // Polygamma function of order n
    static double polygamma(int n, double x);
    
    // Specific implementations for efficiency
    static double digamma(double x);    // n = 0
    static double trigamma(double x);   // n = 1
    static double tetragamma(double x); // n = 2
    static double pentagamma(double x); // n = 3
    
private:
    // Helper for general polygamma using series expansion
    static double polygammaSeries(int n, double x);
    
    // Helper for asymptotic expansion
    static double polygammaAsymptotic(int n, double x);
};