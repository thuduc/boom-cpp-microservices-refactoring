#include "api/glm_handler.hpp"
#include "core/glm_base.hpp"
#include "core/models/linear_regression.hpp"
#include "core/models/logistic_regression.hpp"
#include "core/models/poisson_regression.hpp"
#include "core/models/probit_regression.hpp"
#include "core/diagnostics/regression_diagnostics.hpp"
#include "utils/data_preprocessor.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>
#include <memory>

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

// Create GLM model based on type
std::unique_ptr<GLMBase> createGLM(const std::string& model_type) {
    if (model_type == "linear" || model_type == "regression") {
        return std::make_unique<LinearRegression>();
    } else if (model_type == "logistic") {
        return std::make_unique<LogisticRegression>();
    } else if (model_type == "poisson") {
        return std::make_unique<PoissonRegression>();
    } else if (model_type == "probit") {
        return std::make_unique<ProbitRegression>();
    }
    return nullptr;
}

http::response<http::string_body> handle_glm_request(
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
            {"service", "glm-service"},
            {"version", "1.0.0"},
            {"supported_models", {"linear", "logistic", "poisson", "probit"}}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Fit GLM endpoint pattern: /glm/{model_type}/fit
    std::regex fit_regex("/glm/([a-z]+)/fit");
    std::smatch match;
    if (std::regex_match(target, match, fit_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("X") || !body.contains("y")) {
                return make_error_response(http::status::bad_request, 
                    "Missing X (features) or y (response) data", version, keep_alive);
            }
            
            // Parse data
            Eigen::MatrixXd X = DataPreprocessor::jsonToMatrix(body["X"]);
            Eigen::VectorXd y = DataPreprocessor::jsonToVector(body["y"]);
            
            // Optional parameters
            bool fit_intercept = body.value("fit_intercept", true);
            bool standardize = body.value("standardize", false);
            double regularization = body.value("regularization", 0.0);
            std::string reg_type = body.value("regularization_type", "l2");
            
            // Create model
            auto model = createGLM(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Configure model
            model->setFitIntercept(fit_intercept);
            model->setStandardize(standardize);
            model->setRegularization(regularization, reg_type);
            
            // Optional weights
            if (body.contains("weights")) {
                Eigen::VectorXd weights = DataPreprocessor::jsonToVector(body["weights"]);
                model->setWeights(weights);
            }
            
            // Fit model
            auto result = model->fit(X, y);
            
            json response = {
                {"model_type", model_type},
                {"coefficients", DataPreprocessor::vectorToJson(result.coefficients)},
                {"intercept", result.intercept},
                {"iterations", result.iterations},
                {"converged", result.converged},
                {"log_likelihood", result.log_likelihood},
                {"aic", result.aic},
                {"bic", result.bic}
            };
            
            if (result.standard_errors.size() > 0) {
                response["standard_errors"] = DataPreprocessor::vectorToJson(result.standard_errors);
                response["p_values"] = DataPreprocessor::vectorToJson(result.p_values);
                response["confidence_intervals"] = result.confidence_intervals;
            }
            
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Predict endpoint pattern: /glm/{model_type}/predict
    std::regex predict_regex("/glm/([a-z]+)/predict");
    if (std::regex_match(target, match, predict_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("X") || !body.contains("coefficients")) {
                return make_error_response(http::status::bad_request, 
                    "Missing X (features) or coefficients", version, keep_alive);
            }
            
            // Parse data
            Eigen::MatrixXd X = DataPreprocessor::jsonToMatrix(body["X"]);
            Eigen::VectorXd coefficients = DataPreprocessor::jsonToVector(body["coefficients"]);
            double intercept = body.value("intercept", 0.0);
            
            // Create model
            auto model = createGLM(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Set parameters
            model->setCoefficients(coefficients, intercept);
            
            // Predict
            std::string predict_type = body.value("type", "response");
            Eigen::VectorXd predictions;
            
            if (predict_type == "response") {
                predictions = model->predict(X);
            } else if (predict_type == "linear") {
                predictions = model->predictLinear(X);
            } else if (predict_type == "probability" && 
                      (model_type == "logistic" || model_type == "probit")) {
                predictions = model->predictProbability(X);
            } else {
                return make_error_response(http::status::bad_request, 
                    "Invalid prediction type: " + predict_type, version, keep_alive);
            }
            
            json response = {
                {"model_type", model_type},
                {"predictions", DataPreprocessor::vectorToJson(predictions)},
                {"type", predict_type}
            };
            
            // Add prediction intervals if requested
            if (body.value("prediction_intervals", false)) {
                double alpha = body.value("alpha", 0.05);
                auto intervals = model->predictionIntervals(X, alpha);
                response["prediction_intervals"] = intervals;
            }
            
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Diagnostics endpoint pattern: /glm/{model_type}/diagnostics
    std::regex diag_regex("/glm/([a-z]+)/diagnostics");
    if (std::regex_match(target, match, diag_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("X") || !body.contains("y") || 
                !body.contains("coefficients")) {
                return make_error_response(http::status::bad_request, 
                    "Missing required data", version, keep_alive);
            }
            
            // Parse data
            Eigen::MatrixXd X = DataPreprocessor::jsonToMatrix(body["X"]);
            Eigen::VectorXd y = DataPreprocessor::jsonToVector(body["y"]);
            Eigen::VectorXd coefficients = DataPreprocessor::jsonToVector(body["coefficients"]);
            double intercept = body.value("intercept", 0.0);
            
            // Create model
            auto model = createGLM(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Set parameters
            model->setCoefficients(coefficients, intercept);
            
            // Compute diagnostics
            RegressionDiagnostics diag;
            auto diag_result = diag.compute(model.get(), X, y);
            
            json response = {
                {"model_type", model_type},
                {"residuals", {
                    {"raw", DataPreprocessor::vectorToJson(diag_result.residuals)},
                    {"standardized", DataPreprocessor::vectorToJson(diag_result.standardized_residuals)},
                    {"deviance", DataPreprocessor::vectorToJson(diag_result.deviance_residuals)}
                }},
                {"goodness_of_fit", {
                    {"r_squared", diag_result.r_squared},
                    {"adjusted_r_squared", diag_result.adjusted_r_squared},
                    {"deviance", diag_result.deviance},
                    {"pearson_chi_squared", diag_result.pearson_chi_squared}
                }},
                {"influence", {
                    {"leverage", DataPreprocessor::vectorToJson(diag_result.leverage)},
                    {"cooks_distance", DataPreprocessor::vectorToJson(diag_result.cooks_distance)}
                }},
                {"tests", {
                    {"durbin_watson", diag_result.durbin_watson},
                    {"breusch_pagan_p_value", diag_result.breusch_pagan_p_value},
                    {"shapiro_wilk_p_value", diag_result.shapiro_wilk_p_value}
                }}
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