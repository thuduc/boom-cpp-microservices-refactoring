#ifndef ELLIPTIC_FUNCTIONS_HPP
#define ELLIPTIC_FUNCTIONS_HPP

class EllipticFunctions {
public:
    // Complete elliptic integral of the first kind K(k)
    static double ellipticK(double k);
    
    // Complete elliptic integral of the second kind E(k)
    static double ellipticE(double k);
    
    // Incomplete elliptic integral of the first kind F(phi, k)
    static double ellipticF(double phi, double k);
    
    // Incomplete elliptic integral of the second kind E(phi, k)
    static double ellipticE(double phi, double k);
    
    // Jacobi elliptic functions
    static void jacobiElliptic(double u, double k, double& sn, double& cn, double& dn);
    
private:
    // Arithmetic-geometric mean
    static double agm(double a, double b);
    
    // Landen transformation
    static double landenTransform(double k);
};