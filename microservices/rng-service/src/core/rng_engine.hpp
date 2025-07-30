#ifndef RNG_ENGINE_HPP
#define RNG_ENGINE_HPP

#include <vector>
#include <memory>
#include <mutex>
#include <random>

// Singleton RNG engine wrapper
class RNGEngine {
public:
    static RNGEngine& getInstance();
    
    // Seed the RNG
    void seed(uint64_t seed);
    
    // Generate uniform random numbers
    std::vector<double> uniform(int n, double min = 0.0, double max = 1.0);
    
    // Generate normal random numbers
    std::vector<double> normal(int n, double mean = 0.0, double sd = 1.0);
    
    // Generate gamma random numbers
    std::vector<double> gamma(int n, double shape, double scale);
    
    // Generate beta random numbers
    std::vector<double> beta(int n, double a, double b);
    
    // Generate multinomial random numbers
    std::vector<std::vector<int>> multinomial(int n, int trials, 
                                               const std::vector<double>& probs);

private:
    RNGEngine();
    RNGEngine(const RNGEngine&) = delete;
    RNGEngine& operator=(const RNGEngine&) = delete;
    
    std::mt19937_64 generator_;
    mutable std::mutex mutex_;
};

#endif // RNG_ENGINE_HPP