#ifndef BESSEL_FUNCTIONS_HPP
#define BESSEL_FUNCTIONS_HPP

class BesselFunctions {
public:
    // Bessel functions of the first kind
    static double besselJ(int n, double x);
    
    // Bessel functions of the second kind
    static double besselY(int n, double x);
    
    // Modified Bessel functions of the first kind
    static double besselI(int n, double x);
    
    // Modified Bessel functions of the second kind
    static double besselK(int n, double x);
    
private:
    // Helper functions for series expansions
    static double besselJ0(double x);
    static double besselJ1(double x);
    static double besselY0(double x);
    static double besselY1(double x);
    
    // Recurrence relations
    static double besselJn(int n, double x);
    static double besselYn(int n, double x);
};