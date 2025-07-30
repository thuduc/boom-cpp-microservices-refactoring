#include "hypothesis_tests.hpp"
#include "descriptive_stats.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

TestResult HypothesisTests::oneSampleTTest(const std::vector<double>& data,
                                          double mu,
                                          const std::string& alternative) {
    if (data.size() < 2) {
        throw std::invalid_argument("Need at least 2 observations for t-test");
    }
    
    double mean = DescriptiveStats::mean(data);
    double std_dev = DescriptiveStats::standardDeviation(data);
    double n = data.size();
    double se = std_dev / std::sqrt(n);
    
    TestResult result;
    result.statistic = (mean - mu) / se;
    result.df = n - 1;
    
    // Calculate p-value based on alternative hypothesis
    if (alternative == "two-sided") {
        result.p_value = 2 * (1 - tDistributionCDF(std::abs(result.statistic), result.df));
    } else if (alternative == "less") {
        result.p_value = tDistributionCDF(result.statistic, result.df);
    } else if (alternative == "greater") {
        result.p_value = 1 - tDistributionCDF(result.statistic, result.df);
    }
    
    // Confidence interval (95%)
    double t_critical = 1.96;  // Approximation for large samples
    result.confidence_interval = {
        mean - t_critical * se,
        mean + t_critical * se
    };
    
    result.reject_null = result.p_value < 0.05;
    result.effect_size = (mean - mu) / std_dev;  // Cohen's d
    
    return result;
}

TestResult HypothesisTests::twoSampleTTest(const std::vector<double>& data1,
                                          const std::vector<double>& data2,
                                          bool equal_variance,
                                          const std::string& alternative) {
    if (data1.size() < 2 || data2.size() < 2) {
        throw std::invalid_argument("Need at least 2 observations in each group");
    }
    
    double mean1 = DescriptiveStats::mean(data1);
    double mean2 = DescriptiveStats::mean(data2);
    double var1 = DescriptiveStats::variance(data1);
    double var2 = DescriptiveStats::variance(data2);
    double n1 = data1.size();
    double n2 = data2.size();
    
    TestResult result;
    
    if (equal_variance) {
        // Pooled variance
        double sp = std::sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
        double se = sp * std::sqrt(1/n1 + 1/n2);
        result.statistic = (mean1 - mean2) / se;
        result.df = n1 + n2 - 2;
    } else {
        // Welch's t-test
        double se = std::sqrt(var1/n1 + var2/n2);
        result.statistic = (mean1 - mean2) / se;
        
        // Welch-Satterthwaite degrees of freedom
        double num = std::pow(var1/n1 + var2/n2, 2);
        double denom = std::pow(var1/n1, 2)/(n1 - 1) + std::pow(var2/n2, 2)/(n2 - 1);
        result.df = num / denom;
    }
    
    // Calculate p-value
    if (alternative == "two-sided") {
        result.p_value = 2 * (1 - tDistributionCDF(std::abs(result.statistic), result.df));
    } else if (alternative == "less") {
        result.p_value = tDistributionCDF(result.statistic, result.df);
    } else if (alternative == "greater") {
        result.p_value = 1 - tDistributionCDF(result.statistic, result.df);
    }
    
    result.reject_null = result.p_value < 0.05;
    result.effect_size = mean1 - mean2;  // Mean difference
    
    return result;
}

TestResult HypothesisTests::pairedTTest(const std::vector<double>& data1,
                                       const std::vector<double>& data2,
                                       const std::string& alternative) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("Paired data must have equal length");
    }
    
    // Compute differences
    std::vector<double> differences;
    for (size_t i = 0; i < data1.size(); ++i) {
        differences.push_back(data1[i] - data2[i]);
    }
    
    // Perform one-sample t-test on differences
    return oneSampleTTest(differences, 0.0, alternative);
}

TestResult HypothesisTests::chiSquareTest(const std::vector<std::vector<double>>& observed) {
    if (observed.empty() || observed[0].empty()) {
        throw std::invalid_argument("Contingency table cannot be empty");
    }
    
    size_t rows = observed.size();
    size_t cols = observed[0].size();
    
    // Calculate row and column totals
    std::vector<double> row_totals(rows, 0.0);
    std::vector<double> col_totals(cols, 0.0);
    double grand_total = 0.0;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            row_totals[i] += observed[i][j];
            col_totals[j] += observed[i][j];
            grand_total += observed[i][j];
        }
    }
    
    // Calculate chi-square statistic
    double chi_square = 0.0;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double expected = row_totals[i] * col_totals[j] / grand_total;
            if (expected > 0) {
                double diff = observed[i][j] - expected;
                chi_square += diff * diff / expected;
            }
        }
    }
    
    TestResult result;
    result.statistic = chi_square;
    result.df = (rows - 1) * (cols - 1);
    result.p_value = 1 - chiSquareCDF(chi_square, result.df);
    result.reject_null = result.p_value < 0.05;
    
    return result;
}

TestResult HypothesisTests::oneWayANOVA(const std::vector<std::vector<double>>& groups) {
    if (groups.size() < 2) {
        throw std::invalid_argument("Need at least 2 groups for ANOVA");
    }
    
    // Calculate group means and overall mean
    std::vector<double> group_means;
    std::vector<size_t> group_sizes;
    double grand_mean = 0.0;
    size_t total_n = 0;
    
    for (const auto& group : groups) {
        if (group.empty()) continue;
        double mean = DescriptiveStats::mean(group);
        group_means.push_back(mean);
        group_sizes.push_back(group.size());
        grand_mean += mean * group.size();
        total_n += group.size();
    }
    
    grand_mean /= total_n;
    
    // Calculate between-group sum of squares
    double ss_between = 0.0;
    for (size_t i = 0; i < groups.size(); ++i) {
        double diff = group_means[i] - grand_mean;
        ss_between += group_sizes[i] * diff * diff;
    }
    
    // Calculate within-group sum of squares
    double ss_within = 0.0;
    for (size_t i = 0; i < groups.size(); ++i) {
        for (double value : groups[i]) {
            double diff = value - group_means[i];
            ss_within += diff * diff;
        }
    }
    
    // Degrees of freedom
    double df_between = groups.size() - 1;
    double df_within = total_n - groups.size();
    
    // Mean squares
    double ms_between = ss_between / df_between;
    double ms_within = ss_within / df_within;
    
    TestResult result;
    result.statistic = ms_between / ms_within;  // F-statistic
    result.df = df_between;
    result.df2 = df_within;
    result.p_value = 1 - fDistributionCDF(result.statistic, result.df, result.df2);
    result.reject_null = result.p_value < 0.05;
    
    return result;
}

TestResult HypothesisTests::wilcoxonSignedRank(const std::vector<double>& data1,
                                              const std::vector<double>& data2) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("Paired data must have equal length");
    }
    
    // Calculate differences and ranks
    std::vector<std::pair<double, double>> abs_diff_ranks;
    
    for (size_t i = 0; i < data1.size(); ++i) {
        double diff = data1[i] - data2[i];
        if (diff != 0) {  // Ignore zero differences
            abs_diff_ranks.push_back({std::abs(diff), diff});
        }
    }
    
    // Sort by absolute difference
    std::sort(abs_diff_ranks.begin(), abs_diff_ranks.end());
    
    // Assign ranks and calculate test statistic
    double w_plus = 0.0;
    double w_minus = 0.0;
    
    for (size_t i = 0; i < abs_diff_ranks.size(); ++i) {
        double rank = i + 1;
        if (abs_diff_ranks[i].second > 0) {
            w_plus += rank;
        } else {
            w_minus += rank;
        }
    }
    
    TestResult result;
    result.statistic = std::min(w_plus, w_minus);
    
    // For large samples, use normal approximation
    double n = abs_diff_ranks.size();
    double mean = n * (n + 1) / 4.0;
    double var = n * (n + 1) * (2 * n + 1) / 24.0;
    double z = (result.statistic - mean) / std::sqrt(var);
    
    result.p_value = 2 * normalCDF(z);  // Two-sided test
    result.reject_null = result.p_value < 0.05;
    
    return result;
}

TestResult HypothesisTests::wilcoxonRankSum(const std::vector<double>& data1,
                                           const std::vector<double>& data2) {
    // Mann-Whitney U test
    size_t n1 = data1.size();
    size_t n2 = data2.size();
    
    // Combine and rank data
    std::vector<std::pair<double, int>> combined;
    
    for (double value : data1) {
        combined.push_back({value, 1});
    }
    for (double value : data2) {
        combined.push_back({value, 2});
    }
    
    std::sort(combined.begin(), combined.end());
    
    // Calculate rank sums
    double r1 = 0.0;
    
    for (size_t i = 0; i < combined.size(); ++i) {
        double rank = i + 1;
        if (combined[i].second == 1) {
            r1 += rank;
        }
    }
    
    // Calculate U statistic
    double u1 = r1 - n1 * (n1 + 1) / 2.0;
    double u2 = n1 * n2 - u1;
    
    TestResult result;
    result.statistic = std::min(u1, u2);
    
    // Normal approximation for large samples
    double mean = n1 * n2 / 2.0;
    double var = n1 * n2 * (n1 + n2 + 1) / 12.0;
    double z = (result.statistic - mean) / std::sqrt(var);
    
    result.p_value = 2 * normalCDF(z);
    result.reject_null = result.p_value < 0.05;
    
    return result;
}

// Simplified distribution CDFs (in practice, use better approximations)
double HypothesisTests::tDistributionCDF(double t, double df) {
    // Simplified approximation using normal distribution for large df
    if (df > 30) {
        return normalCDF(t);
    }
    
    // For small df, use approximation
    double x = df / (df + t * t);
    double a = df / 2.0;
    double b = 0.5;
    
    // Incomplete beta function approximation
    return 0.5 + (t > 0 ? 0.5 : -0.5) * (1 - x);
}

double HypothesisTests::chiSquareCDF(double chi2, double df) {
    // Simplified approximation
    // For large df, use normal approximation
    if (df > 30) {
        double z = (std::pow(chi2/df, 1.0/3.0) - (1 - 2.0/(9*df))) / std::sqrt(2.0/(9*df));
        return normalCDF(z);
    }
    
    // Simple approximation for small df
    return 1 - std::exp(-chi2/2);
}

double HypothesisTests::fDistributionCDF(double f, double df1, double df2) {
    // Simplified approximation
    double x = df1 * f / (df1 * f + df2);
    return x;  // Very simplified
}

double HypothesisTests::normalCDF(double z) {
    // Approximation of standard normal CDF
    return 0.5 * (1 + std::erf(z / std::sqrt(2)));
}