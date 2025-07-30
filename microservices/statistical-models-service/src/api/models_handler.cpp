#include "api/models_handler.hpp"
#include "core/model_factory.hpp"
#include "utils/data_validator.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>

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

http::response<http::string_body> handle_models_request(
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
            {"service", "statistical-models-service"},
            {"version", "1.0.0"},
            {"supported_models", {"gaussian", "gamma", "beta", "poisson", "multinomial"}}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Fit model endpoint pattern: /models/{model_type}/fit
    std::regex fit_regex("/models/([a-z]+)/fit");
    std::smatch match;
    if (std::regex_match(target, match, fit_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data")) {
                return make_error_response(http::status::bad_request, 
                    "Missing data field", version, keep_alive);
            }
            
            // Validate data
            std::vector<double> data = body["data"];
            if (!DataValidator::validateData(data, model_type)) {
                return make_error_response(http::status::bad_request, 
                    "Invalid data for model type", version, keep_alive);
            }
            
            // Get estimation method
            std::string method = body.value("method", "mle");
            
            // Create and fit model
            auto model = ModelFactory::createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Fit model
            FitResult fit_result;
            if (method == "mle") {
                fit_result = model->fitMLE(data);
            } else if (method == "bayesian") {
                // Get prior parameters if provided
                json prior_params = body.value("prior", json::object());
                fit_result = model->fitBayesian(data, prior_params);
            } else {
                return make_error_response(http::status::bad_request, 
                    "Unknown estimation method: " + method, version, keep_alive);
            }
            
            json response = {
                {"model_type", model_type},
                {"method", method},
                {"parameters", fit_result.parameters},
                {"log_likelihood", fit_result.log_likelihood},
                {"convergence", {
                    {"converged", fit_result.converged},
                    {"iterations", fit_result.iterations}
                }}
            };
            
            if (fit_result.standard_errors.size() > 0) {
                response["standard_errors"] = fit_result.standard_errors;
            }
            
            if (fit_result.confidence_intervals.size() > 0) {
                response["confidence_intervals"] = fit_result.confidence_intervals;
            }
            
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Predict endpoint pattern: /models/{model_type}/predict
    std::regex predict_regex("/models/([a-z]+)/predict");
    if (std::regex_match(target, match, predict_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("parameters")) {
                return make_error_response(http::status::bad_request, 
                    "Missing parameters field", version, keep_alive);
            }
            
            auto model = ModelFactory::createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Set parameters
            json params = body["parameters"];
            model->setParameters(params);
            
            // Get prediction type
            std::string pred_type = body.value("type", "mean");
            int n_samples = body.value("n_samples", 1);
            
            json response = {
                {"model_type", model_type},
                {"parameters", params}
            };
            
            if (pred_type == "mean") {
                response["prediction"] = model->mean();
            } else if (pred_type == "variance") {
                response["prediction"] = model->variance();
            } else if (pred_type == "quantile") {
                double p = body.value("p", 0.5);
                response["prediction"] = model->quantile(p);
                response["quantile"] = p;
            } else if (pred_type == "pdf") {
                std::vector<double> x = body["x"];
                std::vector<double> pdf_values;
                for (double xi : x) {
                    pdf_values.push_back(model->pdf(xi));
                }
                response["x"] = x;
                response["pdf"] = pdf_values;
            } else if (pred_type == "cdf") {
                std::vector<double> x = body["x"];
                std::vector<double> cdf_values;
                for (double xi : x) {
                    cdf_values.push_back(model->cdf(xi));
                }
                response["x"] = x;
                response["cdf"] = cdf_values;
            }
            
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Simulate endpoint pattern: /models/{model_type}/simulate
    std::regex simulate_regex("/models/([a-z]+)/simulate");
    if (std::regex_match(target, match, simulate_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("parameters")) {
                return make_error_response(http::status::bad_request, 
                    "Missing parameters field", version, keep_alive);
            }
            
            auto model = ModelFactory::createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Set parameters
            json params = body["parameters"];
            model->setParameters(params);
            
            // Generate samples
            int n_samples = body.value("n_samples", 100);
            auto samples = model->simulate(n_samples);
            
            json response = {
                {"model_type", model_type},
                {"parameters", params},
                {"n_samples", n_samples},
                {"samples", samples}
            };
            
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Likelihood endpoint pattern: /models/{model_type}/likelihood
    std::regex likelihood_regex("/models/([a-z]+)/likelihood");
    if (std::regex_match(target, match, likelihood_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("parameters") || !body.contains("data")) {
                return make_error_response(http::status::bad_request, 
                    "Missing parameters or data field", version, keep_alive);
            }
            
            auto model = ModelFactory::createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Set parameters
            json params = body["parameters"];
            model->setParameters(params);
            
            // Compute likelihood
            std::vector<double> data = body["data"];
            double log_likelihood = model->logLikelihood(data);
            
            json response = {
                {"model_type", model_type},
                {"parameters", params},
                {"log_likelihood", log_likelihood},
                {"likelihood", std::exp(log_likelihood)},
                {"n_observations", data.size()}
            };
            
            // Add AIC and BIC
            int n_params = model->getNumParameters();
            int n_obs = data.size();
            response["aic"] = 2 * n_params - 2 * log_likelihood;
            response["bic"] = n_params * std::log(n_obs) - 2 * log_likelihood;
            
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