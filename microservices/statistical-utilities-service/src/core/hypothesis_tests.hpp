#ifndef HYPOTHESIS_TESTS_HPP
#define HYPOTHESIS_TESTS_HPP

#include <vector>
#include <string>

struct TestResult {
    double statistic;
    double p_value;
    double df;
    double df2;  // For F-tests
    bool reject_null;
    double effect_size;
    std::vector<double> confidence_interval;
};

class HypothesisTests {
public:
    // T-tests
    static TestResult oneSampleTTest(const std::vector<double>& data, 
                                    double mu,
                                    const std::string& alternative = "two-sided");
    
    static TestResult twoSampleTTest(const std::vector<double>& data1,
                                    const std::vector<double>& data2,
                                    bool equal_variance = true,
                                    const std::string& alternative = "two-sided");
    
    static TestResult pairedTTest(const std::vector<double>& data1,
                                 const std::vector<double>& data2,
                                 const std::string& alternative = "two-sided");
    
    // Chi-square tests
    static TestResult chiSquareTest(const std::vector<std::vector<double>>& observed);
    static TestResult chiSquareGOF(const std::vector<double>& observed,
                                  const std::vector<double>& expected);
    
    // ANOVA
    static TestResult oneWayANOVA(const std::vector<std::vector<double>>& groups);
    
    // Non-parametric tests
    static TestResult wilcoxonSignedRank(const std::vector<double>& data1,
                                        const std::vector<double>& data2);
    
    static TestResult wilcoxonRankSum(const std::vector<double>& data1,
                                     const std::vector<double>& data2);
    
    static TestResult kruskalWallis(const std::vector<std::vector<double>>& groups);
    
private:
    // Helper functions
    static double tDistributionCDF(double t, double df);
    static double chiSquareCDF(double chi2, double df);
    static double fDistributionCDF(double f, double df1, double df2);
    static double normalCDF(double z);
};