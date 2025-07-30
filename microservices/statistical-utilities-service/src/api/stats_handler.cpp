#include "stats_handler.hpp"
#include "core/descriptive_stats.hpp"
#include "core/hypothesis_tests.hpp"
#include "core/correlation_analysis.hpp"
#include "core/resampling_methods.hpp"
#include "core/distribution_fitting.hpp"
#include "core/time_series_analysis.hpp"
#include "core/multivariate_analysis.hpp"
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

// Helper to create error response
http::response<http::string_body> make_error_response(
    http::status status,
    const std::string& message,
    unsigned version,
    bool keep_alive) {
    
    json error_json = {
        {"error", message},
        {"status", static_cast<int>(status)}
    };
    
    http::response<http::string_body> res{status, version};
    res.set(http::field::content_type, "application/json");
    res.keep_alive(keep_alive);
    res.body() = error_json.dump();
    res.prepare_payload();
    return res;
}

// Helper to create success response
http::response<http::string_body> make_success_response(
    const json& result,
    unsigned version,
    bool keep_alive) {
    
    http::response<http::string_body> res{http::status::ok, version};
    res.set(http::field::content_type, "application/json");
    res.keep_alive(keep_alive);
    res.body() = result.dump();
    res.prepare_payload();
    return res;
}

http::response<http::string_body> handle_stats_request(
    http::request<http::string_body>&& req) {
    
    // Extract basic info
    auto const version = req.version();
    auto const keep_alive = req.keep_alive();
    
    // Handle different endpoints
    std::string target = std::string(req.target());
    
    // Health check endpoint
    if (target == "/health" && req.method() == http::verb::get) {
        json health = {
            {"status", "healthy"},
            {"service", "statistical-utilities-service"},
            {"version", "1.0.0"},
            {"capabilities", {
                "descriptive_statistics",
                "hypothesis_testing",
                "correlation_analysis",
                "resampling_methods",
                "distribution_fitting",
                "time_series_analysis",
                "multivariate_analysis"
            }}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Descriptive statistics
    if (target == "/stats/summary" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data")) {
                return make_error_response(http::status::bad_request,
                    "Missing 'data' field", version, keep_alive);
            }
            
            std::vector<double> data = body["data"].get<std::vector<double>>();
            std::vector<std::string> statistics = body.value("statistics",
                std::vector<std::string>{"mean", "std", "min", "max"});
            
            json result;
            
            for (const auto& stat : statistics) {
                if (stat == "mean") {
                    result["mean"] = DescriptiveStats::mean(data);
                } else if (stat == "median") {
                    result["median"] = DescriptiveStats::median(data);
                } else if (stat == "mode") {
                    result["mode"] = DescriptiveStats::mode(data);
                } else if (stat == "std") {
                    result["std"] = DescriptiveStats::standardDeviation(data);
                } else if (stat == "variance") {
                    result["variance"] = DescriptiveStats::variance(data);
                } else if (stat == "skewness") {
                    result["skewness"] = DescriptiveStats::skewness(data);
                } else if (stat == "kurtosis") {
                    result["kurtosis"] = DescriptiveStats::kurtosis(data);
                } else if (stat == "min") {
                    result["min"] = DescriptiveStats::min(data);
                } else if (stat == "max") {
                    result["max"] = DescriptiveStats::max(data);
                } else if (stat == "range") {
                    result["range"] = DescriptiveStats::range(data);
                } else if (stat == "quantiles") {
                    std::vector<double> probs = body.value("probabilities",
                        std::vector<double>{0.25, 0.5, 0.75});
                    result["quantiles"] = DescriptiveStats::quantiles(data, probs);
                }
            }
            
            result["n"] = data.size();
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Hypothesis testing
    if (target == "/stats/hypothesis-test" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::string test = body["test"];
            json result;
            
            if (test == "t-test") {
                std::string type = body.value("type", "two-sample");
                
                if (type == "one-sample") {
                    std::vector<double> data = body["data1"].get<std::vector<double>>();
                    double mu = body["mu"];
                    std::string alternative = body.value("alternative", "two-sided");
                    
                    auto test_result = HypothesisTests::oneSampleTTest(data, mu, alternative);
                    
                    result = {
                        {"test", "one-sample-t-test"},
                        {"statistic", test_result.statistic},
                        {"p_value", test_result.p_value},
                        {"df", test_result.df},
                        {"reject_null", test_result.reject_null},
                        {"confidence_interval", test_result.confidence_interval}
                    };
                } else if (type == "two-sample") {
                    std::vector<double> data1 = body["data1"].get<std::vector<double>>();
                    std::vector<double> data2 = body["data2"].get<std::vector<double>>();
                    bool equal_var = body.value("equal_variance", true);
                    std::string alternative = body.value("alternative", "two-sided");
                    
                    auto test_result = HypothesisTests::twoSampleTTest(
                        data1, data2, equal_var, alternative);
                    
                    result = {
                        {"test", "two-sample-t-test"},
                        {"statistic", test_result.statistic},
                        {"p_value", test_result.p_value},
                        {"df", test_result.df},
                        {"reject_null", test_result.reject_null},
                        {"mean_difference", test_result.effect_size}
                    };
                } else if (type == "paired") {
                    std::vector<double> data1 = body["data1"].get<std::vector<double>>();
                    std::vector<double> data2 = body["data2"].get<std::vector<double>>();
                    std::string alternative = body.value("alternative", "two-sided");
                    
                    auto test_result = HypothesisTests::pairedTTest(data1, data2, alternative);
                    
                    result = {
                        {"test", "paired-t-test"},
                        {"statistic", test_result.statistic},
                        {"p_value", test_result.p_value},
                        {"df", test_result.df},
                        {"reject_null", test_result.reject_null}
                    };
                }
            } else if (test == "chi-square") {
                std::vector<std::vector<double>> observed = 
                    body["observed"].get<std::vector<std::vector<double>>>();
                
                auto test_result = HypothesisTests::chiSquareTest(observed);
                
                result = {
                    {"test", "chi-square-test"},
                    {"statistic", test_result.statistic},
                    {"p_value", test_result.p_value},
                    {"df", test_result.df},
                    {"reject_null", test_result.reject_null}
                };
            } else if (test == "anova") {
                std::vector<std::vector<double>> groups = 
                    body["groups"].get<std::vector<std::vector<double>>>();
                
                auto test_result = HypothesisTests::oneWayANOVA(groups);
                
                result = {
                    {"test", "one-way-anova"},
                    {"f_statistic", test_result.statistic},
                    {"p_value", test_result.p_value},
                    {"df_between", test_result.df},
                    {"df_within", test_result.df2},
                    {"reject_null", test_result.reject_null}
                };
            } else if (test == "wilcoxon") {
                std::vector<double> data1 = body["data1"].get<std::vector<double>>();
                std::string type = body.value("type", "rank-sum");
                
                if (type == "signed-rank") {
                    std::vector<double> data2 = body["data2"].get<std::vector<double>>();
                    auto test_result = HypothesisTests::wilcoxonSignedRank(data1, data2);
                    
                    result = {
                        {"test", "wilcoxon-signed-rank"},
                        {"statistic", test_result.statistic},
                        {"p_value", test_result.p_value},
                        {"reject_null", test_result.reject_null}
                    };
                } else {
                    std::vector<double> data2 = body["data2"].get<std::vector<double>>();
                    auto test_result = HypothesisTests::wilcoxonRankSum(data1, data2);
                    
                    result = {
                        {"test", "wilcoxon-rank-sum"},
                        {"statistic", test_result.statistic},
                        {"p_value", test_result.p_value},
                        {"reject_null", test_result.reject_null}
                    };
                }
            }
            
            double alpha = body.value("alpha", 0.05);
            result["alpha"] = alpha;
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Correlation analysis
    if (target == "/stats/correlation" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> x = body["x"].get<std::vector<double>>();
            std::vector<double> y = body["y"].get<std::vector<double>>();
            std::string method = body.value("method", "pearson");
            
            CorrelationResult corr_result;
            
            if (method == "pearson") {
                corr_result = CorrelationAnalysis::pearson(x, y);
            } else if (method == "spearman") {
                corr_result = CorrelationAnalysis::spearman(x, y);
            } else if (method == "kendall") {
                corr_result = CorrelationAnalysis::kendall(x, y);
            } else {
                return make_error_response(http::status::bad_request,
                    "Unknown correlation method", version, keep_alive);
            }
            
            json result = {
                {"method", method},
                {"correlation", corr_result.correlation},
                {"p_value", corr_result.p_value},
                {"confidence_interval", corr_result.confidence_interval},
                {"n", x.size()}
            };
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Resampling methods
    if (target == "/stats/resample" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> data = body["data"].get<std::vector<double>>();
            std::string method = body.value("method", "bootstrap");
            int n_samples = body.value("n_samples", 1000);
            std::string statistic = body.value("statistic", "mean");
            
            ResamplingResult resample_result;
            
            if (method == "bootstrap") {
                resample_result = ResamplingMethods::bootstrap(
                    data, n_samples, statistic);
            } else if (method == "jackknife") {
                resample_result = ResamplingMethods::jackknife(data, statistic);
            } else if (method == "permutation") {
                std::vector<double> data2 = body["data2"].get<std::vector<double>>();
                resample_result = ResamplingMethods::permutationTest(
                    data, data2, n_samples, statistic);
            } else {
                return make_error_response(http::status::bad_request,
                    "Unknown resampling method", version, keep_alive);
            }
            
            json result = {
                {"method", method},
                {"statistic", statistic},
                {"estimate", resample_result.estimate},
                {"standard_error", resample_result.standard_error},
                {"confidence_interval", resample_result.confidence_interval},
                {"bias", resample_result.bias}
            };
            
            if (body.value("return_samples", false)) {
                result["samples"] = resample_result.samples;
            }
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Distribution fitting
    if (target == "/stats/fit-distribution" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> data = body["data"].get<std::vector<double>>();
            std::vector<std::string> distributions = body.value("distributions",
                std::vector<std::string>{"normal", "exponential"});
            
            json results = json::array();
            
            for (const auto& dist : distributions) {
                DistributionFitResult fit_result;
                
                if (dist == "normal") {
                    fit_result = DistributionFitting::fitNormal(data);
                } else if (dist == "exponential") {
                    fit_result = DistributionFitting::fitExponential(data);
                } else if (dist == "gamma") {
                    fit_result = DistributionFitting::fitGamma(data);
                } else if (dist == "beta") {
                    fit_result = DistributionFitting::fitBeta(data);
                } else if (dist == "weibull") {
                    fit_result = DistributionFitting::fitWeibull(data);
                } else {
                    continue;
                }
                
                json dist_result = {
                    {"distribution", dist},
                    {"parameters", fit_result.parameters},
                    {"log_likelihood", fit_result.log_likelihood},
                    {"aic", fit_result.aic},
                    {"bic", fit_result.bic},
                    {"ks_statistic", fit_result.ks_statistic},
                    {"ks_p_value", fit_result.ks_p_value}
                };
                
                results.push_back(dist_result);
            }
            
            // Sort by AIC
            std::sort(results.begin(), results.end(),
                [](const json& a, const json& b) {
                    return a["aic"] < b["aic"];
                });
            
            json result = {
                {"best_fit", results[0]},
                {"all_fits", results}
            };
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Time series analysis
    if (target == "/stats/time-series" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> data = body["data"].get<std::vector<double>>();
            std::vector<std::string> analyses = body.value("analysis",
                std::vector<std::string>{"acf", "trend"});
            
            json result;
            
            for (const auto& analysis : analyses) {
                if (analysis == "acf") {
                    int lags = body.value("lags", 20);
                    result["acf"] = TimeSeriesAnalysis::autocorrelation(data, lags);
                } else if (analysis == "pacf") {
                    int lags = body.value("lags", 20);
                    result["pacf"] = TimeSeriesAnalysis::partialAutocorrelation(data, lags);
                } else if (analysis == "trend") {
                    auto trend_result = TimeSeriesAnalysis::detrend(data);
                    result["trend"] = {
                        {"slope", trend_result.slope},
                        {"intercept", trend_result.intercept},
                        {"r_squared", trend_result.r_squared},
                        {"detrended", trend_result.detrended}
                    };
                } else if (analysis == "seasonality") {
                    int period = body.value("period", 12);
                    auto seasonal = TimeSeriesAnalysis::seasonalDecomposition(data, period);
                    result["seasonality"] = {
                        {"seasonal", seasonal.seasonal},
                        {"trend", seasonal.trend},
                        {"residual", seasonal.residual}
                    };
                } else if (analysis == "stationarity") {
                    auto adf_result = TimeSeriesAnalysis::augmentedDickeyFuller(data);
                    result["stationarity"] = {
                        {"adf_statistic", adf_result.statistic},
                        {"p_value", adf_result.p_value},
                        {"is_stationary", adf_result.is_stationary}
                    };
                }
            }
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Principal Component Analysis
    if (target == "/stats/pca" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<std::vector<double>> data = 
                body["data"].get<std::vector<std::vector<double>>>();
            int n_components = body.value("n_components", 2);
            bool standardize = body.value("standardize", true);
            
            auto pca_result = MultivariateAnalysis::pca(data, n_components, standardize);
            
            json result = {
                {"explained_variance", pca_result.explained_variance},
                {"explained_variance_ratio", pca_result.explained_variance_ratio},
                {"cumulative_variance_ratio", pca_result.cumulative_variance_ratio},
                {"components", pca_result.components},
                {"transformed_data", pca_result.transformed_data}
            };
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // ECDF
    if (target == "/stats/ecdf" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> data = body["data"].get<std::vector<double>>();
            std::vector<double> eval_points = body.value("eval_points", data);
            
            auto ecdf_values = DescriptiveStats::ecdf(data, eval_points);
            
            json result = {
                {"eval_points", eval_points},
                {"ecdf_values", ecdf_values}
            };
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Spline fitting
    if (target == "/stats/spline/fit" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> x = body["x"].get<std::vector<double>>();
            std::vector<double> y = body["y"].get<std::vector<double>>();
            int degree = body.value("degree", 3);
            double smoothing = body.value("smoothing", 0.0);
            
            auto spline = DescriptiveStats::fitSpline(x, y, degree, smoothing);
            
            // Evaluate at points
            std::vector<double> eval_x = body.value("eval_x", x);
            std::vector<double> eval_y;
            
            for (double xi : eval_x) {
                eval_y.push_back(spline.evaluate(xi));
            }
            
            json result = {
                {"degree", degree},
                {"smoothing", smoothing},
                {"eval_x", eval_x},
                {"eval_y", eval_y},
                {"knots", spline.getKnots()}
            };
            
            return make_success_response(result, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Not found
    return make_error_response(http::status::not_found,
        "Endpoint not found", version, keep_alive);
}