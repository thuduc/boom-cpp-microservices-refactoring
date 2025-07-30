#include "api/ts_handler.hpp"
#include "core/models/arima_model.hpp"
#include "core/models/local_level.hpp"
#include "core/models/local_linear_trend.hpp"
#include "core/models/seasonal.hpp"
#include "core/filters/kalman_filter.hpp"
#include "utils/time_series_utils.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>
#include <memory>
#include <chrono>

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

// Create time series model based on type
std::unique_ptr<TimeSeriesModel> createModel(const std::string& model_type) {
    if (model_type == "arima") {
        return std::make_unique<ARIMAModel>();
    } else if (model_type == "local_level") {
        return std::make_unique<LocalLevelModel>();
    } else if (model_type == "local_linear_trend") {
        return std::make_unique<LocalLinearTrendModel>();
    } else if (model_type == "seasonal") {
        return std::make_unique<SeasonalModel>();
    }
    return nullptr;
}

http::response<http::string_body> handle_ts_request(
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
            {"service", "time-series-service"},
            {"version", "1.0.0"},
            {"supported_models", {"arima", "local_level", "local_linear_trend", "seasonal"}},
            {"features", {"filtering", "smoothing", "forecasting", "decomposition"}}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Fit model endpoint: /timeseries/{model_type}/fit
    std::regex fit_regex("/timeseries/([a-z_]+)/fit");
    std::smatch match;
    if (std::regex_match(target, match, fit_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'data' field", version, keep_alive);
            }
            
            // Parse time series data
            std::vector<double> data = TimeSeriesUtils::jsonToVector(body["data"]);
            
            // Optional timestamps
            std::vector<double> timestamps;
            if (body.contains("timestamps")) {
                timestamps = TimeSeriesUtils::jsonToVector(body["timestamps"]);
            }
            
            // Create and configure model
            auto model = createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Model-specific parameters
            if (model_type == "arima") {
                int p = body.value("p", 1);  // AR order
                int d = body.value("d", 0);  // Differencing order
                int q = body.value("q", 1);  // MA order
                dynamic_cast<ARIMAModel*>(model.get())->setOrders(p, d, q);
            } else if (model_type == "seasonal") {
                int period = body.value("period", 12);
                dynamic_cast<SeasonalModel*>(model.get())->setPeriod(period);
            }
            
            // Fit model
            auto result = model->fit(data, timestamps);
            
            json response = {
                {"model_type", model_type},
                {"parameters", result.parameters},
                {"log_likelihood", result.log_likelihood},
                {"aic", result.aic},
                {"bic", result.bic},
                {"converged", result.converged},
                {"iterations", result.iterations}
            };
            
            if (!result.state_estimates.empty()) {
                response["state_estimates"] = result.state_estimates;
            }
            
            if (!result.residuals.empty()) {
                response["residuals"] = result.residuals;
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Forecast endpoint: /timeseries/{model_type}/forecast
    std::regex forecast_regex("/timeseries/([a-z_]+)/forecast");
    if (std::regex_match(target, match, forecast_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("parameters") || !body.contains("horizon")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'parameters' or 'horizon'", version, keep_alive);
            }
            
            // Create model and set parameters
            auto model = createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            model->setParameters(body["parameters"]);
            
            // Optional historical data for state initialization
            std::vector<double> data;
            if (body.contains("data")) {
                data = TimeSeriesUtils::jsonToVector(body["data"]);
            }
            
            int horizon = body["horizon"];
            double confidence = body.value("confidence", 0.95);
            
            // Generate forecast
            auto forecast = model->forecast(horizon, data, confidence);
            
            json response = {
                {"model_type", model_type},
                {"horizon", horizon},
                {"point_forecast", forecast.point_forecast},
                {"lower_bound", forecast.lower_bound},
                {"upper_bound", forecast.upper_bound},
                {"confidence", confidence}
            };
            
            if (!forecast.forecast_errors.empty()) {
                response["forecast_errors"] = forecast.forecast_errors;
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Filter/smooth endpoint: /timeseries/filter
    if (target == "/timeseries/filter" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data") || !body.contains("model_type")) {
                return make_error_response(http::status::bad_request, 
                    "Missing required fields", version, keep_alive);
            }
            
            std::vector<double> data = TimeSeriesUtils::jsonToVector(body["data"]);
            std::string model_type = body["model_type"];
            std::string method = body.value("method", "kalman");
            
            // Create model
            auto model = createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Set parameters if provided
            if (body.contains("parameters")) {
                model->setParameters(body["parameters"]);
            }
            
            json response;
            
            if (method == "kalman") {
                KalmanFilter filter;
                auto filtered = filter.filter(model.get(), data);
                
                response = {
                    {"method", "kalman"},
                    {"filtered_states", filtered.filtered_states},
                    {"filtered_variances", filtered.filtered_variances},
                    {"predictions", filtered.predictions},
                    {"innovations", filtered.innovations}
                };
                
                if (body.value("smooth", false)) {
                    auto smoothed = filter.smooth(model.get(), data);
                    response["smoothed_states"] = smoothed.smoothed_states;
                    response["smoothed_variances"] = smoothed.smoothed_variances;
                }
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Decomposition endpoint: /timeseries/decompose
    if (target == "/timeseries/decompose" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'data' field", version, keep_alive);
            }
            
            std::vector<double> data = TimeSeriesUtils::jsonToVector(body["data"]);
            std::string method = body.value("method", "stl");
            int period = body.value("period", 0);
            
            json response;
            
            if (method == "stl") {
                // STL decomposition (Seasonal-Trend decomposition using Loess)
                auto decomp = TimeSeriesUtils::stlDecomposition(data, period);
                response = {
                    {"method", "stl"},
                    {"trend", decomp.trend},
                    {"seasonal", decomp.seasonal},
                    {"remainder", decomp.remainder},
                    {"period", period}
                };
            } else if (method == "classical") {
                // Classical decomposition
                bool multiplicative = body.value("multiplicative", false);
                auto decomp = TimeSeriesUtils::classicalDecomposition(data, period, multiplicative);
                response = {
                    {"method", "classical"},
                    {"trend", decomp.trend},
                    {"seasonal", decomp.seasonal},
                    {"remainder", decomp.remainder},
                    {"period", period},
                    {"type", multiplicative ? "multiplicative" : "additive"}
                };
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Diagnostics endpoint: /timeseries/diagnostics
    if (target == "/timeseries/diagnostics" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("residuals")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'residuals' field", version, keep_alive);
            }
            
            std::vector<double> residuals = TimeSeriesUtils::jsonToVector(body["residuals"]);
            int max_lag = body.value("max_lag", 20);
            
            // Compute various diagnostics
            auto acf = TimeSeriesUtils::autocorrelation(residuals, max_lag);
            auto pacf = TimeSeriesUtils::partialAutocorrelation(residuals, max_lag);
            auto ljung_box = TimeSeriesUtils::ljungBoxTest(residuals, max_lag);
            
            json response = {
                {"acf", acf},
                {"pacf", pacf},
                {"ljung_box", {
                    {"statistic", ljung_box.statistic},
                    {"p_value", ljung_box.p_value},
                    {"degrees_of_freedom", ljung_box.df}
                }},
                {"mean", TimeSeriesUtils::mean(residuals)},
                {"variance", TimeSeriesUtils::variance(residuals)},
                {"skewness", TimeSeriesUtils::skewness(residuals)},
                {"kurtosis", TimeSeriesUtils::kurtosis(residuals)}
            };
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Not found
    return make_error_response(http::status::not_found, 
        "Endpoint not found", version, keep_alive);
}