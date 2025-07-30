#ifndef CORRELATION_ANALYSIS_HPP
#define CORRELATION_ANALYSIS_HPP

#include <vector>

struct CorrelationResult {
    double correlation;
    double p_value;
    std::vector<double> confidence_interval;
};

class CorrelationAnalysis {
public:
    // Pearson correlation coefficient
    static CorrelationResult pearson(const std::vector<double>& x,
                                    const std::vector<double>& y);
    
    // Spearman rank correlation
    static CorrelationResult spearman(const std::vector<double>& x,
                                     const std::vector<double>& y);
    
    // Kendall's tau correlation
    static CorrelationResult kendall(const std::vector<double>& x,
                                    const std::vector<double>& y);
    
    // Partial correlation
    static double partialCorrelation(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<double>& z);
    
    // Correlation matrix
    static std::vector<std::vector<double>> correlationMatrix(
        const std::vector<std::vector<double>>& data);
    
private:
    // Helper functions
    static std::vector<double> rankData(const std::vector<double>& data);
    static double fisherZ(double r);
    static double inverseFisherZ(double z);
};