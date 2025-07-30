#include "api/mcmc_handler.hpp"
#include "core/mcmc_engine.hpp"
#include "core/chain_diagnostics.hpp"
#include "models/chain_manager.hpp"
#include "utils/distribution_parser.hpp"
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

http::response<http::string_body> handle_mcmc_request(
    http::request<http::string_body>&& req,
    std::shared_ptr<ChainManager> chain_manager) {
    
    // Extract basic info
    auto const version = req.version();
    auto const keep_alive = req.keep_alive();
    
    // Handle different endpoints
    std::string target = std::string(req.target());
    
    // Health check endpoint
    if (target == "/health" && req.method() == http::verb::get) {
        json health = {
            {"status", "healthy"},
            {"service", "mcmc-service"},
            {"version", "1.0.0"},
            {"active_chains", chain_manager->getActiveChainCount()}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Metropolis-Hastings sampler endpoint
    if (target == "/mcmc/metropolis-hastings" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("target_distribution") || !body.contains("initial_state")) {
                return make_error_response(http::status::bad_request, 
                    "Missing target_distribution or initial_state", version, keep_alive);
            }
            
            // Parse parameters
            auto target_dist = DistributionParser::parseDistribution(body["target_distribution"]);
            std::vector<double> initial_state = body["initial_state"];
            
            MCMCSettings settings;
            settings.num_samples = body.value("num_samples", 1000);
            settings.burn_in = body.value("burn_in", 100);
            settings.thin = body.value("thin", 1);
            settings.adaptation_period = body.value("adaptation_period", 500);
            
            // Proposal distribution settings
            if (body.contains("proposal")) {
                settings.proposal_type = body["proposal"].value("type", "normal");
                settings.proposal_scale = body["proposal"].value("scale", 1.0);
            }
            
            // Create chain
            std::string chain_id = chain_manager->createChain("metropolis-hastings");
            
            // Run sampler asynchronously
            std::thread([chain_manager, chain_id, target_dist, initial_state, settings]() {
                try {
                    MCMCResult result = MCMCEngine::metropolisHastings(
                        target_dist, initial_state, settings);
                    chain_manager->completeChain(chain_id, result);
                } catch (const std::exception& e) {
                    chain_manager->failChain(chain_id, e.what());
                }
            }).detach();
            
            json response = {
                {"chain_id", chain_id},
                {"status", "running"},
                {"method", "metropolis-hastings"},
                {"message", "MCMC chain started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Slice sampler endpoint
    if (target == "/mcmc/slice" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("target_distribution") || !body.contains("initial_state")) {
                return make_error_response(http::status::bad_request, 
                    "Missing target_distribution or initial_state", version, keep_alive);
            }
            
            auto target_dist = DistributionParser::parseDistribution(body["target_distribution"]);
            std::vector<double> initial_state = body["initial_state"];
            
            MCMCSettings settings;
            settings.num_samples = body.value("num_samples", 1000);
            settings.burn_in = body.value("burn_in", 100);
            settings.thin = body.value("thin", 1);
            settings.slice_width = body.value("slice_width", 1.0);
            settings.max_stepping_out = body.value("max_stepping_out", 10);
            
            std::string chain_id = chain_manager->createChain("slice");
            
            std::thread([chain_manager, chain_id, target_dist, initial_state, settings]() {
                try {
                    MCMCResult result = MCMCEngine::sliceSampler(
                        target_dist, initial_state, settings);
                    chain_manager->completeChain(chain_id, result);
                } catch (const std::exception& e) {
                    chain_manager->failChain(chain_id, e.what());
                }
            }).detach();
            
            json response = {
                {"chain_id", chain_id},
                {"status", "running"},
                {"method", "slice"},
                {"message", "MCMC chain started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // ARMS (Adaptive Rejection Metropolis Sampling) endpoint
    if (target == "/mcmc/arms" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("target_distribution") || !body.contains("initial_state")) {
                return make_error_response(http::status::bad_request, 
                    "Missing target_distribution or initial_state", version, keep_alive);
            }
            
            auto target_dist = DistributionParser::parseDistribution(body["target_distribution"]);
            std::vector<double> initial_state = body["initial_state"];
            
            MCMCSettings settings;
            settings.num_samples = body.value("num_samples", 1000);
            settings.burn_in = body.value("burn_in", 100);
            settings.thin = body.value("thin", 1);
            
            // ARMS specific settings
            if (body.contains("bounds")) {
                settings.lower_bounds = body["bounds"]["lower"];
                settings.upper_bounds = body["bounds"]["upper"];
            }
            
            std::string chain_id = chain_manager->createChain("arms");
            
            std::thread([chain_manager, chain_id, target_dist, initial_state, settings]() {
                try {
                    MCMCResult result = MCMCEngine::arms(
                        target_dist, initial_state, settings);
                    chain_manager->completeChain(chain_id, result);
                } catch (const std::exception& e) {
                    chain_manager->failChain(chain_id, e.what());
                }
            }).detach();
            
            json response = {
                {"chain_id", chain_id},
                {"status", "running"},
                {"method", "arms"},
                {"message", "MCMC chain started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Gibbs sampler endpoint
    if (target == "/mcmc/gibbs" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("conditional_distributions") || !body.contains("initial_state")) {
                return make_error_response(http::status::bad_request, 
                    "Missing conditional_distributions or initial_state", version, keep_alive);
            }
            
            // Parse conditional distributions for each dimension
            std::vector<TargetDistribution> conditionals;
            for (const auto& cond : body["conditional_distributions"]) {
                conditionals.push_back(DistributionParser::parseDistribution(cond));
            }
            
            std::vector<double> initial_state = body["initial_state"];
            
            MCMCSettings settings;
            settings.num_samples = body.value("num_samples", 1000);
            settings.burn_in = body.value("burn_in", 100);
            settings.thin = body.value("thin", 1);
            settings.random_scan = body.value("random_scan", false);
            
            std::string chain_id = chain_manager->createChain("gibbs");
            
            std::thread([chain_manager, chain_id, conditionals, initial_state, settings]() {
                try {
                    MCMCResult result = MCMCEngine::gibbs(
                        conditionals, initial_state, settings);
                    chain_manager->completeChain(chain_id, result);
                } catch (const std::exception& e) {
                    chain_manager->failChain(chain_id, e.what());
                }
            }).detach();
            
            json response = {
                {"chain_id", chain_id},
                {"status", "running"},
                {"method", "gibbs"},
                {"message", "MCMC chain started"}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Get chain samples endpoint
    std::regex chain_regex("/mcmc/chain/([a-zA-Z0-9-]+)");
    std::smatch match;
    if (std::regex_match(target, match, chain_regex) && req.method() == http::verb::get) {
        std::string chain_id = match[1];
        
        auto chain_info = chain_manager->getChainInfo(chain_id);
        if (!chain_info.has_value()) {
            return make_error_response(http::status::not_found, 
                "Chain not found", version, keep_alive);
        }
        
        json response = {
            {"chain_id", chain_id},
            {"status", chain_info->status},
            {"method", chain_info->method},
            {"created_at", chain_info->created_at},
            {"updated_at", chain_info->updated_at}
        };
        
        if (chain_info->status == "completed") {
            // Return summary statistics instead of all samples for efficiency
            response["summary"] = {
                {"num_samples", chain_info->result.samples.size()},
                {"acceptance_rate", chain_info->result.acceptance_rate},
                {"effective_sample_size", chain_info->result.effective_sample_size}
            };
            
            // Include first 100 samples as preview
            size_t preview_size = std::min(size_t(100), chain_info->result.samples.size());
            std::vector<std::vector<double>> preview(
                chain_info->result.samples.begin(), 
                chain_info->result.samples.begin() + preview_size);
            response["preview_samples"] = preview;
            
            // Add download link for full data
            response["download_url"] = "/mcmc/chain/" + chain_id + "/download";
        } else if (chain_info->status == "failed") {
            response["error"] = chain_info->error_message;
        }
        
        return make_success_response(response, version, keep_alive);
    }
    
    // Chain diagnostics endpoint
    if (target == "/mcmc/diagnostics" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("chain_id")) {
                return make_error_response(http::status::bad_request, 
                    "Missing chain_id", version, keep_alive);
            }
            
            std::string chain_id = body["chain_id"];
            auto chain_info = chain_manager->getChainInfo(chain_id);
            
            if (!chain_info.has_value() || chain_info->status != "completed") {
                return make_error_response(http::status::bad_request, 
                    "Chain not found or not completed", version, keep_alive);
            }
            
            // Compute diagnostics
            ChainDiagnostics diagnostics;
            auto diag_result = diagnostics.compute(chain_info->result.samples);
            
            json response = {
                {"chain_id", chain_id},
                {"diagnostics", {
                    {"mean", diag_result.mean},
                    {"std_dev", diag_result.std_dev},
                    {"quantiles", diag_result.quantiles},
                    {"autocorrelation", diag_result.autocorrelation},
                    {"effective_sample_size", diag_result.effective_sample_size},
                    {"gelman_rubin", diag_result.gelman_rubin},
                    {"geweke_z", diag_result.geweke_z}
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