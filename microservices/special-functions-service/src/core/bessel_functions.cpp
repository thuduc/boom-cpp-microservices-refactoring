#include "bessel_functions.hpp"
#include <cmath>
#include <stdexcept>

double BesselFunctions::besselJ(int n, double x) {
    if (n < 0) {
        // Use symmetry relation
        return (n % 2 == 0 ? 1 : -1) * besselJ(-n, x);
    }
    
    if (n == 0) return besselJ0(x);
    if (n == 1) return besselJ1(x);
    
    return besselJn(n, x);
}

double BesselFunctions::besselY(int n, double x) {
    if (x <= 0) {
        throw std::domain_error("besselY: x must be positive");
    }
    
    if (n < 0) {
        // Use symmetry relation
        return (n % 2 == 0 ? 1 : -1) * besselY(-n, x);
    }
    
    if (n == 0) return besselY0(x);
    if (n == 1) return besselY1(x);
    
    return besselYn(n, x);
}

double BesselFunctions::besselI(int n, double x) {
    // Modified Bessel function of the first kind
    // Simple approximation for demonstration
    if (n == 0) {
        // Series expansion for small x
        if (std::abs(x) < 3.0) {
            double sum = 1.0;
            double term = 1.0;
            for (int k = 1; k < 20; ++k) {
                term *= (x * x) / (4.0 * k * k);
                sum += term;
            }
            return sum;
        } else {
            // Asymptotic expansion for large x
            return std::exp(x) / std::sqrt(2 * M_PI * x);
        }
    }
    
    // Use recurrence relation for other orders
    if (n < 0) n = -n;  // I_n(x) = I_{-n}(x)
    
    double i0 = besselI(0, x);
    double i1 = x * i0 / 2.0;  // Approximation
    
    if (n == 1) return i1;
    
    // Recurrence
    for (int k = 2; k <= n; ++k) {
        double ik = 2 * (k - 1) * i1 / x - i0;
        i0 = i1;
        i1 = ik;
    }
    
    return i1;
}

double BesselFunctions::besselK(int n, double x) {
    // Modified Bessel function of the second kind
    if (x <= 0) {
        throw std::domain_error("besselK: x must be positive");
    }
    
    // Simple approximation for K_0
    if (n == 0) {
        if (x < 2.0) {
            return -std::log(x / 2.0) * besselI(0, x);
        } else {
            return std::sqrt(M_PI / (2 * x)) * std::exp(-x);
        }
    }
    
    // Use recurrence for other orders
    double k0 = besselK(0, x);
    double k1 = (1.0 / x + k0);  // Approximation
    
    if (n == 1) return k1;
    
    // Recurrence
    for (int k = 2; k <= n; ++k) {
        double kn = 2 * (k - 1) * k1 / x + k0;
        k0 = k1;
        k1 = kn;
    }
    
    return k1;
}

double BesselFunctions::besselJ0(double x) {
    // Bessel function J_0(x)
    double ax = std::abs(x);
    
    if (ax < 8.0) {
        // Series expansion
        double y = x * x;
        return 1.0 - y / 4.0 + y * y / 64.0 - y * y * y / 2304.0 
               + y * y * y * y / 147456.0;
    } else {
        // Asymptotic expansion
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - 0.785398164;
        double p0 = 1.0;
        double p1 = -0.1098628627e-2;
        double p2 = 0.2734510407e-4;
        double q0 = -0.1562499995e-1;
        double q1 = 0.1430488765e-3;
        
        double p = p0 + y * (p1 + y * p2);
        double q = z * (q0 + y * q1);
        
        return std::sqrt(0.636619772 / ax) * (p * std::cos(xx) - q * std::sin(xx));
    }
}

double BesselFunctions::besselJ1(double x) {
    // Bessel function J_1(x)
    double ax = std::abs(x);
    
    if (ax < 8.0) {
        // Series expansion
        double y = x * x;
        double ans = x * (0.5 - y / 16.0 + y * y / 384.0 - y * y * y / 18432.0);
        return ans;
    } else {
        // Asymptotic expansion
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - 2.356194491;
        double p0 = 1.0;
        double p1 = 0.183105e-2;
        double p2 = -0.3516396496e-4;
        double q0 = 0.04687499995;
        double q1 = -0.2002690873e-3;
        
        double p = p0 + y * (p1 + y * p2);
        double q = z * (q0 + y * q1);
        
        double ans = std::sqrt(0.636619772 / ax) * (p * std::cos(xx) - q * std::sin(xx));
        return x < 0.0 ? -ans : ans;
    }
}

double BesselFunctions::besselY0(double x) {
    // Bessel function Y_0(x)
    if (x < 8.0) {
        double j0 = besselJ0(x);
        return (2.0 / M_PI) * (std::log(x / 2.0) * j0 + 
               (-0.07832358 - 0.0420411 * x * x));
    } else {
        double z = 8.0 / x;
        double y = z * z;
        double xx = x - 0.785398164;
        double p0 = 1.0;
        double p1 = -0.1098628627e-2;
        double p2 = 0.2734510407e-4;
        double q0 = -0.1562499995e-1;
        double q1 = 0.1430488765e-3;
        
        double p = p0 + y * (p1 + y * p2);
        double q = z * (q0 + y * q1);
        
        return std::sqrt(0.636619772 / x) * (p * std::sin(xx) + q * std::cos(xx));
    }
}

double BesselFunctions::besselY1(double x) {
    // Bessel function Y_1(x)
    if (x < 8.0) {
        double j1 = besselJ1(x);
        return (2.0 / M_PI) * (std::log(x / 2.0) * j1 - 1.0 / x + 
               x * (0.074 - 0.133 * x * x));
    } else {
        double z = 8.0 / x;
        double y = z * z;
        double xx = x - 2.356194491;
        double p0 = 1.0;
        double p1 = 0.183105e-2;
        double p2 = -0.3516396496e-4;
        double q0 = 0.04687499995;
        double q1 = -0.2002690873e-3;
        
        double p = p0 + y * (p1 + y * p2);
        double q = z * (q0 + y * q1);
        
        return std::sqrt(0.636619772 / x) * (p * std::sin(xx) + q * std::cos(xx));
    }
}

double BesselFunctions::besselJn(int n, double x) {
    // Use recurrence relation
    if (n == 0) return besselJ0(x);
    if (n == 1) return besselJ1(x);
    
    // Forward recurrence for x > n
    if (x > n) {
        double j0 = besselJ0(x);
        double j1 = besselJ1(x);
        
        for (int k = 2; k <= n; ++k) {
            double jk = 2 * (k - 1) * j1 / x - j0;
            j0 = j1;
            j1 = jk;
        }
        
        return j1;
    } else {
        // Backward recurrence for x < n
        double jn = 0.0;
        double jn1 = 1e-10;  // Arbitrary small value
        
        for (int k = 2 * n; k > n; --k) {
            double jk = 2 * k * jn1 / x - jn;
            jn = jn1;
            jn1 = jk;
        }
        
        // Normalize
        double norm = besselJ0(x) / jn1;
        return jn * norm;
    }
}

double BesselFunctions::besselYn(int n, double x) {
    // Use recurrence relation
    if (n == 0) return besselY0(x);
    if (n == 1) return besselY1(x);
    
    double y0 = besselY0(x);
    double y1 = besselY1(x);
    
    for (int k = 2; k <= n; ++k) {
        double yk = 2 * (k - 1) * y1 / x - y0;
        y0 = y1;
        y1 = yk;
    }
    
    return y1;
}