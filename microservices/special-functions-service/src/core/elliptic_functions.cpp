#include "elliptic_functions.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

double EllipticFunctions::ellipticK(double k) {
    if (k < 0.0 || k >= 1.0) {
        throw std::domain_error("ellipticK: k must be in [0, 1)");
    }
    
    // Special cases
    if (k == 0.0) return M_PI / 2.0;
    
    // Use AGM algorithm
    double a = 1.0;
    double b = std::sqrt(1.0 - k * k);
    double c = k;
    
    const double tolerance = 1e-15;
    
    while (c > tolerance) {
        double a_new = 0.5 * (a + b);
        double b_new = std::sqrt(a * b);
        c = 0.5 * (a - b);
        
        a = a_new;
        b = b_new;
    }
    
    return M_PI / (2.0 * a);
}

double EllipticFunctions::ellipticE(double k) {
    if (k < 0.0 || k > 1.0) {
        throw std::domain_error("ellipticE: k must be in [0, 1]");
    }
    
    // Special cases
    if (k == 0.0) return M_PI / 2.0;
    if (k == 1.0) return 1.0;
    
    // Use AGM algorithm with modifications
    double a = 1.0;
    double b = std::sqrt(1.0 - k * k);
    double c = k;
    
    std::vector<double> c_values;
    const double tolerance = 1e-15;
    
    while (c > tolerance) {
        c_values.push_back(c);
        
        double a_new = 0.5 * (a + b);
        double b_new = std::sqrt(a * b);
        c = 0.5 * (a - b);
        
        a = a_new;
        b = b_new;
    }
    
    // Compute E(k)
    double sum = 1.0;
    double power = 0.5;
    
    for (auto it = c_values.rbegin(); it != c_values.rend(); ++it) {
        sum -= power * (*it) * (*it);
        power *= 2.0;
    }
    
    return sum * ellipticK(k);
}

double EllipticFunctions::ellipticF(double phi, double k) {
    if (k < 0.0 || k >= 1.0) {
        throw std::domain_error("ellipticF: k must be in [0, 1)");
    }
    
    // Reduce phi to [-pi, pi]
    while (phi > M_PI) phi -= 2 * M_PI;
    while (phi < -M_PI) phi += 2 * M_PI;
    
    // Special cases
    if (k == 0.0) return phi;
    if (phi == 0.0) return 0.0;
    
    // Use Landen transformation
    double sin_phi = std::sin(phi);
    double cos_phi = std::cos(phi);
    double k_prime = std::sqrt(1.0 - k * k);
    
    // AGM sequence
    std::vector<double> a_seq, b_seq, phi_seq;
    double a = 1.0;
    double b = k_prime;
    
    phi_seq.push_back(phi);
    
    const double tolerance = 1e-15;
    
    while (std::abs(a - b) > tolerance) {
        a_seq.push_back(a);
        b_seq.push_back(b);
        
        double a_new = 0.5 * (a + b);
        double b_new = std::sqrt(a * b);
        
        // Update phi
        double tan_delta = (b / a) * std::tan(phi_seq.back());
        double delta = std::atan(tan_delta);
        phi_seq.push_back(phi_seq.back() + delta);
        
        a = a_new;
        b = b_new;
    }
    
    return phi_seq.back() / a;
}

double EllipticFunctions::ellipticE(double phi, double k) {
    if (k < 0.0 || k > 1.0) {
        throw std::domain_error("ellipticE: k must be in [0, 1]");
    }
    
    // Special cases
    if (k == 0.0) return phi;
    if (k == 1.0) return std::sin(phi);
    if (phi == 0.0) return 0.0;
    
    // Numerical integration using Gauss-Legendre quadrature
    const int n_points = 32;
    double result = 0.0;
    
    // Transform integral to [0, 1] interval
    double t_start = 0.0;
    double t_end = phi;
    
    // Gauss-Legendre nodes and weights (simplified)
    for (int i = 0; i < n_points; ++i) {
        double t = t_start + (t_end - t_start) * (i + 0.5) / n_points;
        double integrand = std::sqrt(1.0 - k * k * std::sin(t) * std::sin(t));
        result += integrand * (t_end - t_start) / n_points;
    }
    
    return result;
}

void EllipticFunctions::jacobiElliptic(double u, double k, double& sn, double& cn, double& dn) {
    if (k < 0.0 || k >= 1.0) {
        throw std::domain_error("jacobiElliptic: k must be in [0, 1)");
    }
    
    // Special cases
    if (k == 0.0) {
        sn = std::sin(u);
        cn = std::cos(u);
        dn = 1.0;
        return;
    }
    
    if (u == 0.0) {
        sn = 0.0;
        cn = 1.0;
        dn = 1.0;
        return;
    }
    
    // Use descending Landen transformation
    std::vector<double> k_seq;
    double k_current = k;
    
    const double tolerance = 1e-15;
    
    while (k_current > tolerance) {
        k_seq.push_back(k_current);
        k_current = (1.0 - std::sqrt(1.0 - k_current * k_current)) / 
                    (1.0 + std::sqrt(1.0 - k_current * k_current));
    }
    
    // Start with trigonometric functions
    double phi = u;
    for (auto it = k_seq.rbegin(); it != k_seq.rend(); ++it) {
        phi *= 1.0 + *it;
    }
    
    sn = std::sin(phi);
    cn = std::cos(phi);
    dn = 1.0;
    
    // Apply ascending transformation
    for (double ki : k_seq) {
        double sn_new = (1.0 + ki) * sn / (1.0 + ki * sn * sn);
        double cn_new = cn * dn / (1.0 + ki * sn * sn);
        double dn_new = (1.0 - ki * sn * sn) / (1.0 + ki * sn * sn);
        
        sn = sn_new;
        cn = cn_new;
        dn = dn_new;
    }
}

double EllipticFunctions::agm(double a, double b) {
    const double tolerance = 1e-15;
    
    while (std::abs(a - b) > tolerance) {
        double a_new = 0.5 * (a + b);
        double b_new = std::sqrt(a * b);
        
        a = a_new;
        b = b_new;
    }
    
    return a;
}

double EllipticFunctions::landenTransform(double k) {
    return (1.0 - std::sqrt(1.0 - k * k)) / (1.0 + std::sqrt(1.0 - k * k));
}