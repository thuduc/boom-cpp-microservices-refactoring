#ifndef FFT_CALCULATOR_HPP
#define FFT_CALCULATOR_HPP

#include <vector>
#include <complex>

class FFTCalculator {
public:
    // Forward FFT
    static std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& x);
    
    // Inverse FFT
    static std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& x);
    
    // Real-valued FFT
    static std::vector<std::complex<double>> rfft(const std::vector<double>& x);
    
    // 2D FFT
    static std::vector<std::vector<std::complex<double>>> fft2d(
        const std::vector<std::vector<std::complex<double>>>& x);
    
private:
    // Cooley-Tukey FFT algorithm
    static void fftRadix2(std::vector<std::complex<double>>& x);
    
    // Bit reversal permutation
    static void bitReverse(std::vector<std::complex<double>>& x);
    
    // Check if size is power of 2
    static bool isPowerOf2(size_t n);
    
    // Pad to next power of 2
    static size_t nextPowerOf2(size_t n);
};