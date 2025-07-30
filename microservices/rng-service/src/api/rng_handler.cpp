#include "api/rng_handler.hpp"
#include "core/rng_engine.hpp"
#include "utils/json_utils.hpp"
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

http::response<http::string_body> handle_request(http::request<http::string_body>&& req) {
    // Extract basic info
    auto const version = req.version();
    auto const keep_alive = req.keep_alive();
    
    // Handle different endpoints
    std::string target = std::string(req.target());
    
    // Health check endpoint
    if (target == "/health" && req.method() == http::verb::get) {
        json health = {
            {"status", "healthy"},
            {"service", "rng-service"},
            {"version", "1.0.0"}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Seed endpoint
    if (target == "/rng/seed" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            if (!body.contains("seed")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'seed' parameter", version, keep_alive);
            }
            
            uint64_t seed = body["seed"];
            RNGEngine::getInstance().seed(seed);
            
            json result = {
                {"status", "success"},
                {"message", "RNG seeded successfully"},
                {"seed", seed}
            };
            return make_success_response(result, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Uniform distribution endpoint
    if (target == "/rng/uniform" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            int n = body.value("n", 1);
            double min = body.value("min", 0.0);
            double max = body.value("max", 1.0);
            
            if (n <= 0 || n > 10000) {
                return make_error_response(http::status::bad_request, 
                    "n must be between 1 and 10000", version, keep_alive);
            }
            
            std::vector<double> samples = RNGEngine::getInstance().uniform(n, min, max);
            
            json result = {
                {"distribution", "uniform"},
                {"parameters", {{"min", min}, {"max", max}}},
                {"n", n},
                {"samples", samples}
            };
            return make_success_response(result, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Normal distribution endpoint
    if (target == "/rng/normal" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            int n = body.value("n", 1);
            double mean = body.value("mean", 0.0);
            double sd = body.value("sd", 1.0);
            
            if (n <= 0 || n > 10000) {
                return make_error_response(http::status::bad_request, 
                    "n must be between 1 and 10000", version, keep_alive);
            }
            
            if (sd <= 0) {
                return make_error_response(http::status::bad_request, 
                    "sd must be positive", version, keep_alive);
            }
            
            std::vector<double> samples = RNGEngine::getInstance().normal(n, mean, sd);
            
            json result = {
                {"distribution", "normal"},
                {"parameters", {{"mean", mean}, {"sd", sd}}},
                {"n", n},
                {"samples", samples}
            };
            return make_success_response(result, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Gamma distribution endpoint
    if (target == "/rng/gamma" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            int n = body.value("n", 1);
            double shape = body.value("shape", 1.0);
            double scale = body.value("scale", 1.0);
            
            if (n <= 0 || n > 10000) {
                return make_error_response(http::status::bad_request, 
                    "n must be between 1 and 10000", version, keep_alive);
            }
            
            if (shape <= 0 || scale <= 0) {
                return make_error_response(http::status::bad_request, 
                    "shape and scale must be positive", version, keep_alive);
            }
            
            std::vector<double> samples = RNGEngine::getInstance().gamma(n, shape, scale);
            
            json result = {
                {"distribution", "gamma"},
                {"parameters", {{"shape", shape}, {"scale", scale}}},
                {"n", n},
                {"samples", samples}
            };
            return make_success_response(result, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Beta distribution endpoint
    if (target == "/rng/beta" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            int n = body.value("n", 1);
            double a = body.value("a", 1.0);
            double b = body.value("b", 1.0);
            
            if (n <= 0 || n > 10000) {
                return make_error_response(http::status::bad_request, 
                    "n must be between 1 and 10000", version, keep_alive);
            }
            
            if (a <= 0 || b <= 0) {
                return make_error_response(http::status::bad_request, 
                    "a and b must be positive", version, keep_alive);
            }
            
            std::vector<double> samples = RNGEngine::getInstance().beta(n, a, b);
            
            json result = {
                {"distribution", "beta"},
                {"parameters", {{"a", a}, {"b", b}}},
                {"n", n},
                {"samples", samples}
            };
            return make_success_response(result, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Multinomial distribution endpoint
    if (target == "/rng/multinomial" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            int n = body.value("n", 1);
            int trials = body.value("trials", 1);
            std::vector<double> probs = body.value("probabilities", std::vector<double>{});
            
            if (n <= 0 || n > 1000) {
                return make_error_response(http::status::bad_request, 
                    "n must be between 1 and 1000", version, keep_alive);
            }
            
            if (trials <= 0) {
                return make_error_response(http::status::bad_request, 
                    "trials must be positive", version, keep_alive);
            }
            
            if (probs.empty()) {
                return make_error_response(http::status::bad_request, 
                    "probabilities array cannot be empty", version, keep_alive);
            }
            
            // Validate probabilities sum to 1
            double sum = 0.0;
            for (double p : probs) {
                if (p < 0) {
                    return make_error_response(http::status::bad_request, 
                        "probabilities must be non-negative", version, keep_alive);
                }
                sum += p;
            }
            
            if (std::abs(sum - 1.0) > 1e-6) {
                return make_error_response(http::status::bad_request, 
                    "probabilities must sum to 1", version, keep_alive);
            }
            
            std::vector<std::vector<int>> samples = 
                RNGEngine::getInstance().multinomial(n, trials, probs);
            
            json result = {
                {"distribution", "multinomial"},
                {"parameters", {{"trials", trials}, {"probabilities", probs}}},
                {"n", n},
                {"samples", samples}
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