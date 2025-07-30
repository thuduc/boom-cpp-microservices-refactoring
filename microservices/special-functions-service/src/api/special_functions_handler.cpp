#include "special_functions_handler.hpp"
#include "core/bessel_functions.hpp"
#include "core/gamma_functions.hpp"
#include "core/beta_functions.hpp"
#include "core/fft_calculator.hpp"
#include "core/polygamma_functions.hpp"
#include "core/hypergeometric_functions.hpp"
#include "core/elliptic_functions.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>
#include <vector>
#include <complex>

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

http::response<http::string_body> handle_special_functions_request(
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
            {"service", "special-functions-service"},
            {"version", "1.0.0"},
            {"functions", {
                "bessel", "gamma", "beta", "fft", "polygamma",
                "hypergeometric", "elliptic"
            }}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Bessel functions
    if (target == "/functions/bessel" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::string type = body.value("type", "J");
            int order = body.value("order", 0);
            double x = body["x"];
            
            double result;
            
            if (type == "J") {
                result = BesselFunctions::besselJ(order, x);
            } else if (type == "Y") {
                result = BesselFunctions::besselY(order, x);
            } else if (type == "I") {
                result = BesselFunctions::besselI(order, x);
            } else if (type == "K") {
                result = BesselFunctions::besselK(order, x);
            } else {
                return make_error_response(http::status::bad_request,
                    "Invalid Bessel function type", version, keep_alive);
            }
            
            json response = {
                {"function", "bessel"},
                {"type", type},
                {"order", order},
                {"x", x},
                {"result", result}
            };
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Gamma functions
    if (target == "/functions/gamma" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::string type = body.value("type", "gamma");
            double x = body["x"];
            
            double result;
            
            if (type == "gamma") {
                result = GammaFunctions::gamma(x);
            } else if (type == "loggamma") {
                result = GammaFunctions::logGamma(x);
            } else if (type == "digamma") {
                result = GammaFunctions::digamma(x);
            } else if (type == "trigamma") {
                result = GammaFunctions::trigamma(x);
            } else {
                return make_error_response(http::status::bad_request,
                    "Invalid gamma function type", version, keep_alive);
            }
            
            json response = {
                {"function", "gamma"},
                {"type", type},
                {"x", x},
                {"result", result}
            };
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Beta functions
    if (target == "/functions/beta" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::string type = body.value("type", "beta");
            double a = body["a"];
            double b = body["b"];
            
            double result;
            
            if (type == "beta") {
                result = BetaFunctions::beta(a, b);
            } else if (type == "logbeta") {
                result = BetaFunctions::logBeta(a, b);
            } else if (type == "regularized") {
                double x = body["x"];
                result = BetaFunctions::regularizedBeta(x, a, b);
            } else {
                return make_error_response(http::status::bad_request,
                    "Invalid beta function type", version, keep_alive);
            }
            
            json response = {
                {"function", "beta"},
                {"type", type},
                {"a", a},
                {"b", b},
                {"result", result}
            };
            
            if (type == "regularized") {
                response["x"] = body["x"];
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // FFT
    if (target == "/functions/fft" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::vector<double> data = body["data"].get<std::vector<double>>();
            bool inverse = body.value("inverse", false);
            
            std::vector<std::complex<double>> complex_data;
            for (double val : data) {
                complex_data.push_back(std::complex<double>(val, 0.0));
            }
            
            std::vector<std::complex<double>> result;
            
            if (inverse) {
                result = FFTCalculator::ifft(complex_data);
            } else {
                result = FFTCalculator::fft(complex_data);
            }
            
            // Convert result to real and imaginary parts
            json real_parts = json::array();
            json imag_parts = json::array();
            
            for (const auto& c : result) {
                real_parts.push_back(c.real());
                imag_parts.push_back(c.imag());
            }
            
            json response = {
                {"function", "fft"},
                {"inverse", inverse},
                {"real", real_parts},
                {"imaginary", imag_parts}
            };
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Polygamma functions
    if (target == "/functions/polygamma" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            int n = body["n"];
            double x = body["x"];
            
            double result = PolygammaFunctions::polygamma(n, x);
            
            json response = {
                {"function", "polygamma"},
                {"n", n},
                {"x", x},
                {"result", result}
            };
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Hypergeometric functions
    if (target == "/functions/hypergeometric" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::string type = body["type"];
            double x = body["x"];
            
            double result;
            
            if (type == "1F1") {
                double a = body["a"];
                double b = body["b"];
                result = HypergeometricFunctions::hypergeometric1F1(a, b, x);
            } else if (type == "2F1") {
                double a = body["a"];
                double b = body["b"];
                double c = body["c"];
                result = HypergeometricFunctions::hypergeometric2F1(a, b, c, x);
            } else {
                return make_error_response(http::status::bad_request,
                    "Invalid hypergeometric function type", version, keep_alive);
            }
            
            json response = {
                {"function", "hypergeometric"},
                {"type", type},
                {"x", x},
                {"result", result}
            };
            
            if (type == "1F1") {
                response["a"] = body["a"];
                response["b"] = body["b"];
            } else {
                response["a"] = body["a"];
                response["b"] = body["b"];
                response["c"] = body["c"];
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Elliptic functions
    if (target == "/functions/elliptic" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            std::string type = body["type"];
            double k = body["k"];  // Modulus
            
            double result;
            
            if (type == "K") {
                result = EllipticFunctions::ellipticK(k);
            } else if (type == "E") {
                result = EllipticFunctions::ellipticE(k);
            } else if (type == "F") {
                double phi = body["phi"];
                result = EllipticFunctions::ellipticF(phi, k);
            } else if (type == "E_incomplete") {
                double phi = body["phi"];
                result = EllipticFunctions::ellipticE(phi, k);
            } else {
                return make_error_response(http::status::bad_request,
                    "Invalid elliptic function type", version, keep_alive);
            }
            
            json response = {
                {"function", "elliptic"},
                {"type", type},
                {"k", k},
                {"result", result}
            };
            
            if (type == "F" || type == "E_incomplete") {
                response["phi"] = body["phi"];
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request,
                e.what(), version, keep_alive);
        }
    }
    
    // Batch evaluation
    if (target == "/functions/batch" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            json results = json::array();
            
            for (const auto& request : body["requests"]) {
                std::string function = request["function"];
                json params = request["params"];
                
                try {
                    json result;
                    
                    if (function == "gamma") {
                        double x = params["x"];
                        std::string type = params.value("type", "gamma");
                        
                        if (type == "gamma") {
                            result["value"] = GammaFunctions::gamma(x);
                        } else if (type == "loggamma") {
                            result["value"] = GammaFunctions::logGamma(x);
                        }
                    } else if (function == "bessel") {
                        std::string type = params["type"];
                        int order = params["order"];
                        double x = params["x"];
                        
                        if (type == "J") {
                            result["value"] = BesselFunctions::besselJ(order, x);
                        } else if (type == "Y") {
                            result["value"] = BesselFunctions::besselY(order, x);
                        }
                    }
                    // Add more functions as needed
                    
                    result["status"] = "success";
                    result["function"] = function;
                    results.push_back(result);
                    
                } catch (const std::exception& e) {
                    json error_result = {
                        {"status", "error"},
                        {"function", function},
                        {"error", e.what()}
                    };
                    results.push_back(error_result);
                }
            }
            
            json response = {
                {"batch_results", results}
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