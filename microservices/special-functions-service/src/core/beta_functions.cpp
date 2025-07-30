#include "beta_functions.hpp"
#include "gamma_functions.hpp"
#include <cmath>
#include <stdexcept>

double BetaFunctions::beta(double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("beta: arguments must be positive");
    }
    
    // B(a,b) = Gamma(a) * Gamma(b) / Gamma(a + b)
    return std::exp(logBeta(a, b));
}

double BetaFunctions::logBeta(double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("logBeta: arguments must be positive");
    }
    
    // log B(a,b) = log Gamma(a) + log Gamma(b) - log Gamma(a + b)
    return GammaFunctions::logGamma(a) + GammaFunctions::logGamma(b) - 
           GammaFunctions::logGamma(a + b);
}

double BetaFunctions::regularizedBeta(double x, double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("regularizedBeta: a and b must be positive");
    }
    
    if (x < 0.0 || x > 1.0) {
        throw std::domain_error("regularizedBeta: x must be in [0, 1]");
    }
    
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    
    // Use symmetry relation if x > (a + 1)/(a + b + 2)
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - regularizedBeta(1.0 - x, b, a);
    }
    
    // Compute using continued fraction
    double bt = std::exp(a * std::log(x) + b * std::log(1.0 - x) - logBeta(a, b));
    
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betaContinuedFraction(x, a, b) / a;
    } else {
        return 1.0 - bt * betaContinuedFraction(1.0 - x, b, a) / b;
    }
}

double BetaFunctions::inverseBeta(double p, double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        throw std::domain_error("inverseBeta: a and b must be positive");
    }
    
    if (p < 0.0 || p > 1.0) {
        throw std::domain_error("inverseBeta: p must be in [0, 1]");
    }
    
    if (p == 0.0) return 0.0;
    if (p == 1.0) return 1.0;
    
    // Initial guess using normal approximation
    double mu = a / (a + b);
    double var = a * b / ((a + b) * (a + b) * (a + b + 1.0));
    double x = mu;
    
    // Newton-Raphson iteration
    for (int iter = 0; iter < 50; ++iter) {
        double f = regularizedBeta(x, a, b) - p;
        
        // Compute derivative (PDF of beta distribution)
        double pdf = std::pow(x, a - 1.0) * std::pow(1.0 - x, b - 1.0) / 
                     std::exp(logBeta(a, b));
        
        double dx = f / pdf;
        x -= dx;
        
        // Keep x in bounds
        if (x <= 0.0) x = 0.5 * (x + dx);
        if (x >= 1.0) x = 0.5 * (x + dx);
        
        if (std::abs(dx) < 1e-10) break;
    }
    
    return x;
}

double BetaFunctions::betaContinuedFraction(double x, double a, double b) {
    // Continued fraction for incomplete beta
    const int max_iterations = 100;
    const double epsilon = 1e-10;
    
    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    
    if (std::abs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    double h = d;
    
    for (int m = 1; m <= max_iterations; ++m) {
        int m2 = 2 * m;
        
        // Even step
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        h *= d * c;
        
        // Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = 1.0 + aa / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        
        if (std::abs(del - 1.0) < epsilon) break;
    }
    
    return h;
}