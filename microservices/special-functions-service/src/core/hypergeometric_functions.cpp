#include "hypergeometric_functions.hpp"
#include "gamma_functions.hpp"
#include <cmath>
#include <stdexcept>

double HypergeometricFunctions::hypergeometric1F1(double a, double b, double x) {
    if (b <= 0.0 && b == std::floor(b)) {
        throw std::domain_error("hypergeometric1F1: b is a non-positive integer");
    }
    
    // Special cases
    if (a == 0.0) return 1.0;
    if (x == 0.0) return 1.0;
    
    // For large negative x, use Kummer transformation
    if (x < -10.0) {
        return kummerTransform(a, b, x);
    }
    
    // Use series expansion
    return hypergeometric1F1Series(a, b, x);
}

double HypergeometricFunctions::hypergeometric2F1(double a, double b, double c, double x) {
    if (c <= 0.0 && c == std::floor(c)) {
        throw std::domain_error("hypergeometric2F1: c is a non-positive integer");
    }
    
    if (std::abs(x) >= 1.0 && x != 1.0) {
        throw std::domain_error("hypergeometric2F1: |x| must be < 1 for series expansion");
    }
    
    // Special cases
    if (a == 0.0 || b == 0.0) return 1.0;
    if (x == 0.0) return 1.0;
    
    // Special case x = 1 (requires c > a + b)
    if (x == 1.0) {
        if (c <= a + b) {
            throw std::domain_error("hypergeometric2F1: divergent at x=1");
        }
        return GammaFunctions::gamma(c) * GammaFunctions::gamma(c - a - b) /
               (GammaFunctions::gamma(c - a) * GammaFunctions::gamma(c - b));
    }
    
    // Use series expansion
    return hypergeometric2F1Series(a, b, c, x);
}

double HypergeometricFunctions::hypergeometricPFQ(const double* a, int p, 
                                                  const double* b, int q, double x) {
    // Check convergence
    if (!checkConvergence(p, q, x)) {
        throw std::domain_error("hypergeometricPFQ: series does not converge");
    }
    
    // Check for poles
    for (int i = 0; i < q; ++i) {
        if (b[i] <= 0.0 && b[i] == std::floor(b[i])) {
            throw std::domain_error("hypergeometricPFQ: pole at non-positive integer");
        }
    }
    
    // Series expansion
    double sum = 1.0;
    double term = 1.0;
    const int max_terms = 500;
    const double tolerance = 1e-12;
    
    for (int n = 1; n < max_terms; ++n) {
        // Compute (a_1)_n * (a_2)_n * ... * (a_p)_n
        double num = 1.0;
        for (int i = 0; i < p; ++i) {
            num *= (a[i] + n - 1);
        }
        
        // Compute (b_1)_n * (b_2)_n * ... * (b_q)_n
        double den = 1.0;
        for (int i = 0; i < q; ++i) {
            den *= (b[i] + n - 1);
        }
        
        // Update term
        term *= num * x / (den * n);
        sum += term;
        
        // Check convergence
        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }
    
    return sum;
}

double HypergeometricFunctions::hypergeometric1F1Series(double a, double b, double x) {
    double sum = 1.0;
    double term = 1.0;
    const int max_terms = 500;
    const double tolerance = 1e-12;
    
    for (int n = 1; n < max_terms; ++n) {
        term *= a * x / (b * n);
        sum += term;
        
        // Update parameters for next term
        a += 1.0;
        b += 1.0;
        
        // Check convergence
        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }
    
    return sum;
}

double HypergeometricFunctions::hypergeometric2F1Series(double a, double b, double c, double x) {
    double sum = 1.0;
    double term = 1.0;
    const int max_terms = 500;
    const double tolerance = 1e-12;
    
    for (int n = 1; n < max_terms; ++n) {
        term *= a * b * x / (c * n);
        sum += term;
        
        // Update parameters for next term
        a += 1.0;
        b += 1.0;
        c += 1.0;
        
        // Check convergence
        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }
    
    return sum;
}

double HypergeometricFunctions::kummerTransform(double a, double b, double x) {
    // Kummer's transformation: 1F1(a; b; x) = exp(x) * 1F1(b-a; b; -x)
    return std::exp(x) * hypergeometric1F1Series(b - a, b, -x);
}

bool HypergeometricFunctions::checkConvergence(int p, int q, double x) {
    // pFq converges for all finite x if p <= q
    if (p <= q) return true;
    
    // pFq converges for |x| < 1 if p = q + 1
    if (p == q + 1 && std::abs(x) < 1.0) return true;
    
    // Otherwise diverges
    return false;
}