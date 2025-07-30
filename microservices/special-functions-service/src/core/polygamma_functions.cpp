#include "polygamma_functions.hpp"
#include "gamma_functions.hpp"
#include <cmath>
#include <stdexcept>

double PolygammaFunctions::polygamma(int n, double x) {
    if (n < 0) {
        throw std::domain_error("polygamma: order must be non-negative");
    }
    
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("polygamma: pole at non-positive integer");
    }
    
    // Use specific implementations for low orders
    if (n == 0) return digamma(x);
    if (n == 1) return trigamma(x);
    if (n == 2) return tetragamma(x);
    if (n == 3) return pentagamma(x);
    
    // For higher orders, use general formula
    return polygammaSeries(n, x);
}

double PolygammaFunctions::digamma(double x) {
    // Delegate to GammaFunctions implementation
    return GammaFunctions::digamma(x);
}

double PolygammaFunctions::trigamma(double x) {
    // Delegate to GammaFunctions implementation
    return GammaFunctions::trigamma(x);
}

double PolygammaFunctions::tetragamma(double x) {
    // psi''(x) - third derivative of log gamma
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("tetragamma: pole at non-positive integer");
    }
    
    // For negative x, use reflection formula
    if (x < 0.0) {
        double pi_x = M_PI * x;
        double sin_pi_x = std::sin(pi_x);
        double cos_pi_x = std::cos(pi_x);
        return -tetragamma(1.0 - x) + 2 * M_PI * M_PI * M_PI * 
               cos_pi_x / (sin_pi_x * sin_pi_x * sin_pi_x);
    }
    
    // For small x, use recurrence
    double result = 0.0;
    while (x < 7.0) {
        result -= 2.0 / (x * x * x);
        x += 1.0;
    }
    
    // Asymptotic expansion
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double inv_x3 = inv_x2 * inv_x;
    double inv_x4 = inv_x2 * inv_x2;
    double inv_x6 = inv_x3 * inv_x3;
    
    result += -2.0 * inv_x3 - inv_x4 - inv_x4 * inv_x / 3.0;
    result += inv_x6 * inv_x / 15.0;
    
    return result;
}

double PolygammaFunctions::pentagamma(double x) {
    // psi'''(x) - fourth derivative of log gamma
    if (x <= 0.0 && x == std::floor(x)) {
        throw std::domain_error("pentagamma: pole at non-positive integer");
    }
    
    // For negative x, use reflection formula
    if (x < 0.0) {
        double pi_x = M_PI * x;
        double sin_pi_x = std::sin(pi_x);
        double cos_pi_x = std::cos(pi_x);
        double pi4 = M_PI * M_PI * M_PI * M_PI;
        
        return pentagamma(1.0 - x) + pi4 * (1.0 + 2.0 * cos_pi_x * cos_pi_x) / 
               (sin_pi_x * sin_pi_x * sin_pi_x * sin_pi_x);
    }
    
    // For small x, use recurrence
    double result = 0.0;
    while (x < 7.0) {
        result += 6.0 / (x * x * x * x);
        x += 1.0;
    }
    
    // Asymptotic expansion
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double inv_x4 = inv_x2 * inv_x2;
    double inv_x5 = inv_x4 * inv_x;
    double inv_x6 = inv_x4 * inv_x2;
    double inv_x8 = inv_x4 * inv_x4;
    
    result += 6.0 * inv_x4 + 3.0 * inv_x5 + inv_x6;
    result -= 0.4 * inv_x8;
    
    return result;
}

double PolygammaFunctions::polygammaSeries(int n, double x) {
    // General polygamma function using series expansion
    if (x < 10.0) {
        // Use recurrence to shift to larger x
        double result = 0.0;
        double sign = (n % 2 == 0) ? -1.0 : 1.0;
        
        while (x < 10.0) {
            double term = 1.0;
            for (int k = 0; k <= n; ++k) {
                term *= (k == 0) ? 1.0 : (n - k + 1);
            }
            result += sign * term / std::pow(x, n + 1);
            x += 1.0;
        }
        
        return result + polygammaAsymptotic(n, x);
    } else {
        return polygammaAsymptotic(n, x);
    }
}

double PolygammaFunctions::polygammaAsymptotic(int n, double x) {
    // Asymptotic expansion for large x
    double sign = (n % 2 == 0) ? -1.0 : 1.0;
    double factorial = 1.0;
    
    for (int k = 1; k <= n; ++k) {
        factorial *= k;
    }
    
    double result = sign * factorial / std::pow(x, n + 1);
    
    // Add correction terms
    double inv_x2 = 1.0 / (x * x);
    double term = result * (n + 1) / (2.0 * x);
    result += term;
    
    // Bernoulli number corrections (simplified)
    if (n <= 3) {
        term *= inv_x2 * (n + 1) * (n + 2) / 12.0;
        result += term;
    }
    
    return result;
}