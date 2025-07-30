#include "resampling_methods.hpp"
#include "descriptive_stats.hpp"
#include <random>
#include <algorithm>
#include <cmath>

ResamplingResult ResamplingMethods::bootstrap(const std::vector<double>& data,
                                             int n_samples,
                                             const std::string& statistic) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);
    
    ResamplingResult result;
    result.samples.reserve(n_samples);
    
    // Original statistic
    double original_stat = computeStatistic(data, statistic);
    
    // Bootstrap samples
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(data.size());
        for (size_t j = 0; j < data.size(); ++j) {
            sample[j] = data[dis(gen)];
        }
        
        double stat = computeStatistic(sample, statistic);
        result.samples.push_back(stat);
    }
    
    // Calculate results
    result.estimate = original_stat;
    
    // Standard error
    double mean = DescriptiveStats::mean(result.samples);
    double sum_sq = 0.0;
    for (double s : result.samples) {
        sum_sq += (s - mean) * (s - mean);
    }
    result.standard_error = std::sqrt(sum_sq / (n_samples - 1));
    
    // Bias
    result.bias = mean - original_stat;
    
    // Confidence interval (percentile method)
    std::vector<double> sorted_samples = result.samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    
    int lower_idx = std::floor(0.025 * n_samples);
    int upper_idx = std::ceil(0.975 * n_samples) - 1;
    
    result.confidence_interval = {
        sorted_samples[lower_idx],
        sorted_samples[upper_idx]
    };
    
    return result;
}

ResamplingResult ResamplingMethods::jackknife(const std::vector<double>& data,
                                             const std::string& statistic) {
    size_t n = data.size();
    ResamplingResult result;
    result.samples.reserve(n);
    
    // Original statistic
    double original_stat = computeStatistic(data, statistic);
    
    // Jackknife samples (leave-one-out)
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> sample;
        sample.reserve(n - 1);
        
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                sample.push_back(data[j]);
            }
        }
        
        double stat = computeStatistic(sample, statistic);
        result.samples.push_back(stat);
    }
    
    // Calculate results
    double mean_jackknife = DescriptiveStats::mean(result.samples);
    
    // Pseudovalues
    std::vector<double> pseudovalues;
    for (double jack_stat : result.samples) {
        pseudovalues.push_back(n * original_stat - (n - 1) * jack_stat);
    }
    
    result.estimate = DescriptiveStats::mean(pseudovalues);
    result.bias = (n - 1) * (mean_jackknife - original_stat);
    
    // Standard error
    double sum_sq = 0.0;
    for (double pv : pseudovalues) {
        sum_sq += (pv - result.estimate) * (pv - result.estimate);
    }
    result.standard_error = std::sqrt(sum_sq / (n * (n - 1)));
    
    // Confidence interval
    double t_critical = 1.96;  // Approximation
    result.confidence_interval = {
        result.estimate - t_critical * result.standard_error,
        result.estimate + t_critical * result.standard_error
    };
    
    return result;
}

ResamplingResult ResamplingMethods::permutationTest(const std::vector<double>& data1,
                                                   const std::vector<double>& data2,
                                                   int n_permutations,
                                                   const std::string& statistic) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    ResamplingResult result;
    result.samples.reserve(n_permutations);
    
    // Combine data
    std::vector<double> combined;
    combined.insert(combined.end(), data1.begin(), data1.end());
    combined.insert(combined.end(), data2.begin(), data2.end());
    
    // Original statistic
    double original_stat = computeStatistic(data1, data2, statistic);
    
    // Permutation samples
    int extreme_count = 0;
    
    for (int i = 0; i < n_permutations; ++i) {
        // Shuffle combined data
        std::shuffle(combined.begin(), combined.end(), gen);
        
        // Split into two groups
        std::vector<double> perm1(combined.begin(), combined.begin() + data1.size());
        std::vector<double> perm2(combined.begin() + data1.size(), combined.end());
        
        double stat = computeStatistic(perm1, perm2, statistic);
        result.samples.push_back(stat);
        
        if (std::abs(stat) >= std::abs(original_stat)) {
            extreme_count++;
        }
    }
    
    result.estimate = original_stat;
    
    // P-value
    double p_value = static_cast<double>(extreme_count + 1) / (n_permutations + 1);
    
    // Standard error from permutation distribution
    double mean = DescriptiveStats::mean(result.samples);
    double sum_sq = 0.0;
    for (double s : result.samples) {
        sum_sq += (s - mean) * (s - mean);
    }
    result.standard_error = std::sqrt(sum_sq / (n_permutations - 1));
    
    result.bias = 0.0;  // Permutation test is unbiased
    
    // Confidence interval from permutation distribution
    std::vector<double> sorted_samples = result.samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    
    int lower_idx = std::floor(0.025 * n_permutations);
    int upper_idx = std::ceil(0.975 * n_permutations) - 1;
    
    result.confidence_interval = {
        sorted_samples[lower_idx],
        sorted_samples[upper_idx]
    };
    
    return result;
}

std::vector<double> ResamplingMethods::crossValidation(const std::vector<double>& x,
                                                       const std::vector<double>& y,
                                                       int n_folds) {
    size_t n = x.size();
    size_t fold_size = n / n_folds;
    
    std::vector<double> cv_scores;
    
    for (int fold = 0; fold < n_folds; ++fold) {
        size_t test_start = fold * fold_size;
        size_t test_end = (fold == n_folds - 1) ? n : (fold + 1) * fold_size;
        
        // Create training and test sets
        std::vector<double> train_x, train_y, test_x, test_y;
        
        for (size_t i = 0; i < n; ++i) {
            if (i >= test_start && i < test_end) {
                test_x.push_back(x[i]);
                test_y.push_back(y[i]);
            } else {
                train_x.push_back(x[i]);
                train_y.push_back(y[i]);
            }
        }
        
        // Simple linear regression on training data
        double mean_x = DescriptiveStats::mean(train_x);
        double mean_y = DescriptiveStats::mean(train_y);
        
        double cov = 0.0;
        double var_x = 0.0;
        
        for (size_t i = 0; i < train_x.size(); ++i) {
            cov += (train_x[i] - mean_x) * (train_y[i] - mean_y);
            var_x += (train_x[i] - mean_x) * (train_x[i] - mean_x);
        }
        
        double slope = cov / var_x;
        double intercept = mean_y - slope * mean_x;
        
        // Calculate MSE on test set
        double mse = 0.0;
        for (size_t i = 0; i < test_x.size(); ++i) {
            double pred = slope * test_x[i] + intercept;
            double error = test_y[i] - pred;
            mse += error * error;
        }
        mse /= test_x.size();
        
        cv_scores.push_back(mse);
    }
    
    return cv_scores;
}

double ResamplingMethods::computeStatistic(const std::vector<double>& data,
                                          const std::string& statistic) {
    if (statistic == "mean") {
        return DescriptiveStats::mean(data);
    } else if (statistic == "median") {
        return DescriptiveStats::median(data);
    } else if (statistic == "std") {
        return DescriptiveStats::standardDeviation(data);
    } else if (statistic == "variance") {
        return DescriptiveStats::variance(data);
    } else if (statistic == "skewness") {
        return DescriptiveStats::skewness(data);
    } else if (statistic == "kurtosis") {
        return DescriptiveStats::kurtosis(data);
    } else {
        return DescriptiveStats::mean(data);  // Default
    }
}

double ResamplingMethods::computeStatistic(const std::vector<double>& data1,
                                          const std::vector<double>& data2,
                                          const std::string& statistic) {
    if (statistic == "mean_diff") {
        return DescriptiveStats::mean(data1) - DescriptiveStats::mean(data2);
    } else if (statistic == "median_diff") {
        return DescriptiveStats::median(data1) - DescriptiveStats::median(data2);
    } else if (statistic == "t_statistic") {
        double mean1 = DescriptiveStats::mean(data1);
        double mean2 = DescriptiveStats::mean(data2);
        double var1 = DescriptiveStats::variance(data1);
        double var2 = DescriptiveStats::variance(data2);
        double n1 = data1.size();
        double n2 = data2.size();
        
        double pooled_se = std::sqrt(var1/n1 + var2/n2);
        return (mean1 - mean2) / pooled_se;
    } else {
        return DescriptiveStats::mean(data1) - DescriptiveStats::mean(data2);  // Default
    }
}