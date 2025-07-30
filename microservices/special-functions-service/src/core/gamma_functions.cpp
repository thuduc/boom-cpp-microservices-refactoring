#include "gamma_functions.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

double GammaFunctions::gamma(double x) {
    // Handle special cases
    if (x == 0.0) {
        throw std::domain_error("gamma: pole at x = 0");
    }
    
    if (x < 0.0 && x == std::floor(x)) {
        throw std::domain_error("gamma: pole at negative integer");
    }
    
    // For negative x, use reflection formula
    if (x < 0.0) {
        return M_PI / (std::sin(M_PI * x) * gamma(1.0 - x));
    }
    
    // For small positive integers, use exact values
    if (x == 1.0) return 1.0;
    if (x == 2.0) return 1.0;
    if (x == 3.0) return 2.0;
    if (x == 4.0) return 6.0;
    
    // Use Lanczos approximation
    return std::exp(logGamma(x));
}

double GammaFunctions::logGamma(double x) {
    if (x <= 0.0) {
        throw std::domain_error("logGamma: x must be positive");
    }
    
    // Lanczos coefficients
    static const double g = 7;
    static const std::vector<double> c = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };
    
    if (x < 0.5) {
        // Use reflection formula
        return std::log(M_PI) - std::log(std::sin(M_PI * x)) - logGamma(1.0 - x);
    }
    
    x -= 1.0;
    double z = x + g + 0.5;
    double series = c[0];
    
    for (int i = 1; i < 9; ++i) {
        series += c[i] / (x + i);
    }
    
    return std::log(std::sqrt(2 * M_PI)) + std::log(series) - z + (x + 0.5) * std::log(z);
}

double GammaFunctions::digamma(double x) {
    // Psi function (derivative of log gamma)
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("digamma: pole at non-positive integer");
    }
    
    // For negative x, use reflection formula
    if (x < 0.0) {
        return digamma(1.0 - x) - M_PI / std::tan(M_PI * x);
    }
    
    // For small x, use recurrence to shift to larger values
    double result = 0.0;
    while (x < 7.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    
    // Asymptotic expansion
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    
    result += std::log(x) - 0.5 * inv_x;
    result -= inv_x2 / 12.0;
    result += inv_x2 * inv_x2 / 120.0;
    result -= inv_x2 * inv_x2 * inv_x2 / 252.0;
    
    return result;
}

double GammaFunctions::trigamma(double x) {
    // Second derivative of log gamma
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("trigamma: pole at non-positive integer");
    }
    
    // For negative x, use reflection formula
    if (x < 0.0) {
        double pi_x = M_PI * x;
        double sin_pi_x = std::sin(pi_x);
        return trigamma(1.0 - x) + M_PI * M_PI / (sin_pi_x * sin_pi_x);
    }
    
    // For small x, use recurrence
    double result = 0.0;
    while (x < 7.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    
    // Asymptotic expansion
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double inv_x3 = inv_x2 * inv_x;
    double inv_x5 = inv_x3 * inv_x2;
    
    result += inv_x + 0.5 * inv_x2 + inv_x3 / 6.0;
    result -= inv_x5 / 30.0;
    result += inv_x5 * inv_x2 / 42.0;
    
    return result;
}

double GammaFunctions::lowerIncompleteGamma(double a, double x) {
    if (a <= 0.0) {
        throw std::domain_error("lowerIncompleteGamma: a must be positive");
    }
    
    if (x < 0.0) {
        throw std::domain_error("lowerIncompleteGamma: x must be non-negative");
    }
    
    if (x == 0.0) return 0.0;
    
    // Use series expansion for small x
    if (x < a + 1.0) {
        double sum = 1.0 / a;
        double term = 1.0 / a;
        
        for (int n = 1; n < 100; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < 1e-10 * std::abs(sum)) break;
        }
        
        return sum * std::pow(x, a) * std::exp(-x);
    } else {
        // Use regularized form and relation
        return gamma(a) * regularizedGammaP(a, x);
    }
}

double GammaFunctions::upperIncompleteGamma(double a, double x) {
    return gamma(a) - lowerIncompleteGamma(a, x);
}

double GammaFunctions::regularizedGammaP(double a, double x) {
    if (a <= 0.0) {
        throw std::domain_error("regularizedGammaP: a must be positive");
    }
    
    if (x < 0.0) {
        throw std::domain_error("regularizedGammaP: x must be non-negative");
    }
    
    if (x == 0.0) return 0.0;
    
    // Use series expansion for x < a + 1
    if (x < a + 1.0) {
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        
        for (int n = 1; n < 100; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-10) break;
        }
        
        return sum * std::exp(-x + a * std::log(x) - logGamma(a));
    } else {
        // Use continued fraction for x >= a + 1
        return 1.0 - regularizedGammaQ(a, x);
    }
}

double GammaFunctions::regularizedGammaQ(double a, double x) {
    if (a <= 0.0) {
        throw std::domain_error("regularizedGammaQ: a must be positive");
    }
    
    if (x < 0.0) {
        throw std::domain_error("regularizedGammaQ: x must be non-negative");
    }
    
    if (x == 0.0) return 1.0;
    
    // Use continued fraction
    double b = x + 1.0 - a;
    double c = 1.0 / 1e-30;
    double d = 1.0 / b;
    double h = d;
    
    for (int i = 1; i < 100; ++i) {
        double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < 1e-30) d = 1e-30;
        c = b + an / c;
        if (std::abs(c) < 1e-30) c = 1e-30;
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-10) break;
    }
    
    return std::exp(-x + a * std::log(x) - logGamma(a)) * h;
}

double GammaFunctions::lanczosApproximation(double z) {
    // Already implemented in logGamma
    return std::exp(logGamma(z));
}

double GammaFunctions::stirlingApproximation(double x) {
    // Stirling's approximation for large x
    return std::sqrt(2 * M_PI / x) * std::pow(x / M_E, x);
}