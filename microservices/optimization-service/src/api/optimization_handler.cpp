#include "api/optimization_handler.hpp"
#include "core/optimization_engine.hpp"
#include "models/job_manager.hpp"
#include "utils/function_parser.hpp"
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

http::response<http::string_body> handle_optimization_request(
    http::request<http::string_body>&& req,
    std::shared_ptr<JobManager> job_manager) {
    
    // Extract basic info
    auto const version = req.version();
    auto const keep_alive = req.keep_alive();
    
    // Handle different endpoints
    std::string target = std::string(req.target());
    
    // Health check endpoint
    if (target == "/health" && req.method() == http::verb::get) {
        json health = {
            {"status", "healthy"},
            {"service", "optimization-service"},
            {"version", "1.0.0"},
            {"active_jobs", job_manager->getActiveJobCount()}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // BFGS optimization endpoint
    if (target == "/optimize/bfgs" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("objective") || !body.contains("initial_point")) {
                return make_error_response(http::status::bad_request, 
                    "Missing objective function or initial_point", version, keep_alive);
            }
            
            // Parse parameters
            auto objective_func = FunctionParser::parseObjective(body["objective"]);
            std::vector<double> initial_point = body["initial_point"];
            
            // Optional parameters
            OptimizationSettings settings;
            settings.max_iterations = body.value("max_iterations", 1000);
            settings.tolerance = body.value("tolerance", 1e-6);
            settings.use_gradient = body.value("use_gradient", false);
            
            if (settings.use_gradient && body.contains("gradient")) {
                settings.gradient_func = FunctionParser::parseGradient(body["gradient"]);
            }
            
            // Create job
            std::string job_id = job_manager->createJob("bfgs");
            
            // Run optimization asynchronously
            std::thread([job_manager, job_id, objective_func, initial_point, settings]() {
                try {
                    OptimizationResult result = OptimizationEngine::bfgs(
                        objective_func, initial_point, settings);
                    job_manager->completeJob(job_id, result);
                } catch (const std::exception& e) {
                    job_manager->failJob(job_id, e.what());
                }
            }).detach();
            
            json response = {
                {"job_id", job_id},
                {"status", "running"},
                {"method", "bfgs"},
                {"message", "Optimization job started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Nelder-Mead optimization endpoint
    if (target == "/optimize/nelder-mead" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("objective") || !body.contains("initial_point")) {
                return make_error_response(http::status::bad_request, 
                    "Missing objective function or initial_point", version, keep_alive);
            }
            
            auto objective_func = FunctionParser::parseObjective(body["objective"]);
            std::vector<double> initial_point = body["initial_point"];
            
            OptimizationSettings settings;
            settings.max_iterations = body.value("max_iterations", 1000);
            settings.tolerance = body.value("tolerance", 1e-6);
            settings.simplex_size = body.value("simplex_size", 0.1);
            
            std::string job_id = job_manager->createJob("nelder-mead");
            
            std::thread([job_manager, job_id, objective_func, initial_point, settings]() {
                try {
                    OptimizationResult result = OptimizationEngine::nelderMead(
                        objective_func, initial_point, settings);
                    job_manager->completeJob(job_id, result);
                } catch (const std::exception& e) {
                    job_manager->failJob(job_id, e.what());
                }
            }).detach();
            
            json response = {
                {"job_id", job_id},
                {"status", "running"},
                {"method", "nelder-mead"},
                {"message", "Optimization job started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Powell optimization endpoint
    if (target == "/optimize/powell" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("objective") || !body.contains("initial_point")) {
                return make_error_response(http::status::bad_request, 
                    "Missing objective function or initial_point", version, keep_alive);
            }
            
            auto objective_func = FunctionParser::parseObjective(body["objective"]);
            std::vector<double> initial_point = body["initial_point"];
            
            OptimizationSettings settings;
            settings.max_iterations = body.value("max_iterations", 1000);
            settings.tolerance = body.value("tolerance", 1e-6);
            
            std::string job_id = job_manager->createJob("powell");
            
            std::thread([job_manager, job_id, objective_func, initial_point, settings]() {
                try {
                    OptimizationResult result = OptimizationEngine::powell(
                        objective_func, initial_point, settings);
                    job_manager->completeJob(job_id, result);
                } catch (const std::exception& e) {
                    job_manager->failJob(job_id, e.what());
                }
            }).detach();
            
            json response = {
                {"job_id", job_id},
                {"status", "running"},
                {"method", "powell"},
                {"message", "Optimization job started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Simulated annealing endpoint
    if (target == "/optimize/simulated-annealing" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("objective") || !body.contains("initial_point")) {
                return make_error_response(http::status::bad_request, 
                    "Missing objective function or initial_point", version, keep_alive);
            }
            
            auto objective_func = FunctionParser::parseObjective(body["objective"]);
            std::vector<double> initial_point = body["initial_point"];
            
            OptimizationSettings settings;
            settings.max_iterations = body.value("max_iterations", 10000);
            settings.initial_temperature = body.value("initial_temperature", 1.0);
            settings.cooling_rate = body.value("cooling_rate", 0.95);
            settings.step_size = body.value("step_size", 0.1);
            
            // Bounds (optional)
            if (body.contains("lower_bounds") && body.contains("upper_bounds")) {
                settings.lower_bounds = body["lower_bounds"].get<std::vector<double>>();
                settings.upper_bounds = body["upper_bounds"].get<std::vector<double>>();
            }
            
            std::string job_id = job_manager->createJob("simulated-annealing");
            
            std::thread([job_manager, job_id, objective_func, initial_point, settings]() {
                try {
                    OptimizationResult result = OptimizationEngine::simulatedAnnealing(
                        objective_func, initial_point, settings);
                    job_manager->completeJob(job_id, result);
                } catch (const std::exception& e) {
                    job_manager->failJob(job_id, e.what());
                }
            }).detach();
            
            json response = {
                {"job_id", job_id},
                {"status", "running"},
                {"method", "simulated-annealing"},
                {"message", "Optimization job started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Newton's method endpoint
    if (target == "/optimize/newton" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("objective") || !body.contains("initial_point")) {
                return make_error_response(http::status::bad_request, 
                    "Missing objective function or initial_point", version, keep_alive);
            }
            
            if (!body.contains("gradient") || !body.contains("hessian")) {
                return make_error_response(http::status::bad_request, 
                    "Newton's method requires gradient and hessian functions", 
                    version, keep_alive);
            }
            
            auto objective_func = FunctionParser::parseObjective(body["objective"]);
            auto gradient_func = FunctionParser::parseGradient(body["gradient"]);
            auto hessian_func = FunctionParser::parseHessian(body["hessian"]);
            std::vector<double> initial_point = body["initial_point"];
            
            OptimizationSettings settings;
            settings.max_iterations = body.value("max_iterations", 100);
            settings.tolerance = body.value("tolerance", 1e-6);
            settings.gradient_func = gradient_func;
            settings.hessian_func = hessian_func;
            
            std::string job_id = job_manager->createJob("newton");
            
            std::thread([job_manager, job_id, objective_func, initial_point, settings]() {
                try {
                    OptimizationResult result = OptimizationEngine::newton(
                        objective_func, initial_point, settings);
                    job_manager->completeJob(job_id, result);
                } catch (const std::exception& e) {
                    job_manager->failJob(job_id, e.what());
                }
            }).detach();
            
            json response = {
                {"job_id", job_id},
                {"status", "running"},
                {"method", "newton"},
                {"message", "Optimization job started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Get job status endpoint
    std::regex status_regex("/optimize/status/([a-zA-Z0-9-]+)");
    std::smatch match;
    if (std::regex_match(target, match, status_regex) && req.method() == http::verb::get) {
        std::string job_id = match[1];
        
        auto job_info = job_manager->getJobInfo(job_id);
        if (!job_info.has_value()) {
            return make_error_response(http::status::not_found, 
                "Job not found", version, keep_alive);
        }
        
        json response = {
            {"job_id", job_id},
            {"status", job_info->status},
            {"method", job_info->method},
            {"created_at", job_info->created_at},
            {"updated_at", job_info->updated_at}
        };
        
        if (job_info->status == "completed") {
            response["result"] = {
                {"optimal_point", job_info->result.optimal_point},
                {"optimal_value", job_info->result.optimal_value},
                {"iterations", job_info->result.iterations},
                {"converged", job_info->result.converged},
                {"convergence_reason", job_info->result.convergence_reason}
            };
            
            if (!job_info->result.gradient.empty()) {
                response["result"]["final_gradient"] = job_info->result.gradient;
            }
        } else if (job_info->status == "failed") {
            response["error"] = job_info->error_message;
        }
        
        return make_success_response(response, version, keep_alive);
    }
    
    // Not found
    return make_error_response(http::status::not_found, 
        "Endpoint not found", version, keep_alive);
}