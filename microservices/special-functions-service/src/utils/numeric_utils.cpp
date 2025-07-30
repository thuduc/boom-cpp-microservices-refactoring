#include "numeric_utils.hpp"
#include <cmath>
#include <algorithm>

double NumericUtils::derivative(double (*f)(double), double x, double h) {
    // Central difference approximation
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

double NumericUtils::integrate(double (*f)(double), double a, double b, int n) {
    // Simpson's rule
    if (n % 2 == 1) n++;  // Ensure even number of intervals
    
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    for (int i = 1; i < n; i += 2) {
        sum += 4.0 * f(a + i * h);
    }
    
    for (int i = 2; i < n; i += 2) {
        sum += 2.0 * f(a + i * h);
    }
    
    return sum * h / 3.0;
}

double NumericUtils::findRoot(double (*f)(double), double a, double b, double tol) {
    // Bisection method
    double fa = f(a);
    double fb = f(b);
    
    if (fa * fb > 0) {
        // Try to find a sign change
        double mid = (a + b) / 2.0;
        double fmid = f(mid);
        
        if (fa * fmid < 0) {
            b = mid;
            fb = fmid;
        } else if (fb * fmid < 0) {
            a = mid;
            fa = fmid;
        } else {
            // No sign change found
            return std::abs(fa) < std::abs(fb) ? a : b;
        }
    }
    
    while (std::abs(b - a) > tol) {
        double c = (a + b) / 2.0;
        double fc = f(c);
        
        if (std::abs(fc) < tol) {
            return c;
        }
        
        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    return (a + b) / 2.0;
}

std::complex<double> NumericUtils::complexPower(const std::complex<double>& z, double n) {
    if (z == std::complex<double>(0.0, 0.0)) {
        return n > 0 ? std::complex<double>(0.0, 0.0) : 
               std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);
    }
    
    double r = std::abs(z);
    double theta = std::arg(z);
    
    double result_r = std::pow(r, n);
    double result_theta = n * theta;
    
    return std::complex<double>(
        result_r * std::cos(result_theta),
        result_r * std::sin(result_theta)
    );
}

std::complex<double> NumericUtils::complexLog(const std::complex<double>& z) {
    return std::log(z);  // Use standard library implementation
}

bool NumericUtils::almostEqual(double a, double b, double tol) {
    return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

double NumericUtils::roundToSignificant(double x, int digits) {
    if (x == 0.0) return 0.0;
    
    double factor = std::pow(10.0, digits - std::ceil(std::log10(std::abs(x))));
    return std::round(x * factor) / factor;
}