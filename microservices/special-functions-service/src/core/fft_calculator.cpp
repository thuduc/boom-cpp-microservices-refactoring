#include "fft_calculator.hpp"
#include <cmath>
#include <algorithm>

std::vector<std::complex<double>> FFTCalculator::fft(const std::vector<std::complex<double>>& x) {
    std::vector<std::complex<double>> result = x;
    
    // Pad to next power of 2 if necessary
    size_t n = result.size();
    size_t n_padded = nextPowerOf2(n);
    
    if (n_padded > n) {
        result.resize(n_padded, std::complex<double>(0.0, 0.0));
    }
    
    // Perform FFT
    fftRadix2(result);
    
    // Trim to original size
    result.resize(n);
    
    return result;
}

std::vector<std::complex<double>> FFTCalculator::ifft(const std::vector<std::complex<double>>& x) {
    std::vector<std::complex<double>> result = x;
    
    // Conjugate input
    for (auto& val : result) {
        val = std::conj(val);
    }
    
    // Forward FFT
    result = fft(result);
    
    // Conjugate output and scale
    double scale = 1.0 / result.size();
    for (auto& val : result) {
        val = std::conj(val) * scale;
    }
    
    return result;
}

std::vector<std::complex<double>> FFTCalculator::rfft(const std::vector<double>& x) {
    std::vector<std::complex<double>> complex_x;
    complex_x.reserve(x.size());
    
    for (double val : x) {
        complex_x.emplace_back(val, 0.0);
    }
    
    return fft(complex_x);
}

std::vector<std::vector<std::complex<double>>> FFTCalculator::fft2d(
    const std::vector<std::vector<std::complex<double>>>& x) {
    
    size_t rows = x.size();
    if (rows == 0) return x;
    
    size_t cols = x[0].size();
    std::vector<std::vector<std::complex<double>>> result = x;
    
    // FFT along rows
    for (size_t i = 0; i < rows; ++i) {
        result[i] = fft(result[i]);
    }
    
    // Transpose
    std::vector<std::vector<std::complex<double>>> transposed(cols, 
        std::vector<std::complex<double>>(rows));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = result[i][j];
        }
    }
    
    // FFT along columns (now rows of transposed)
    for (size_t j = 0; j < cols; ++j) {
        transposed[j] = fft(transposed[j]);
    }
    
    // Transpose back
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = transposed[j][i];
        }
    }
    
    return result;
}

void FFTCalculator::fftRadix2(std::vector<std::complex<double>>& x) {
    size_t n = x.size();
    
    if (n <= 1) return;
    
    // Bit reversal
    bitReverse(x);
    
    // Cooley-Tukey FFT
    for (size_t len = 2; len <= n; len *= 2) {
        double angle = -2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(angle), std::sin(angle));
        
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = x[i + j];
                std::complex<double> v = x[i + j + len / 2] * w;
                
                x[i + j] = u + v;
                x[i + j + len / 2] = u - v;
                
                w *= wlen;
            }
        }
    }
}

void FFTCalculator::bitReverse(std::vector<std::complex<double>>& x) {
    size_t n = x.size();
    size_t bits = 0;
    
    // Count bits
    size_t temp = n - 1;
    while (temp > 0) {
        bits++;
        temp >>= 1;
    }
    
    // Perform bit reversal
    for (size_t i = 0; i < n; ++i) {
        size_t rev = 0;
        size_t num = i;
        
        for (size_t j = 0; j < bits; ++j) {
            rev = (rev << 1) | (num & 1);
            num >>= 1;
        }
        
        if (i < rev) {
            std::swap(x[i], x[rev]);
        }
    }
}

bool FFTCalculator::isPowerOf2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

size_t FFTCalculator::nextPowerOf2(size_t n) {
    if (isPowerOf2(n)) return n;
    
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    
    return power;
}