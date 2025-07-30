#include "api/mixture_handler.hpp"
#include "core/models/gaussian_mixture.hpp"
#include "core/models/dirichlet_process.hpp"
#include "core/models/hierarchical_dp.hpp"
#include "core/samplers/gibbs_sampler.hpp"
#include "core/samplers/collapsed_gibbs.hpp"
#include "utils/clustering_utils.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <regex>
#include <memory>
#include <random>

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

// Create mixture model based on type
std::unique_ptr<MixtureModel> createModel(const std::string& model_type) {
    if (model_type == "gaussian_mixture" || model_type == "gmm") {
        return std::make_unique<GaussianMixture>();
    } else if (model_type == "dirichlet_process" || model_type == "dp") {
        return std::make_unique<DirichletProcess>();
    } else if (model_type == "hierarchical_dp" || model_type == "hdp") {
        return std::make_unique<HierarchicalDP>();
    }
    return nullptr;
}

http::response<http::string_body> handle_mixture_request(
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
            {"service", "mixture-models-service"},
            {"version", "1.0.0"},
            {"supported_models", {"gaussian_mixture", "dirichlet_process", "hierarchical_dp"}},
            {"inference_methods", {"em", "gibbs", "collapsed_gibbs", "variational"}}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Fit mixture model: /mixture/{model_type}/fit
    std::regex fit_regex("/mixture/([a-z_]+)/fit");
    std::smatch match;
    if (std::regex_match(target, match, fit_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'data' field", version, keep_alive);
            }
            
            // Parse data
            Eigen::MatrixXd data = ClusteringUtils::jsonToMatrix(body["data"]);
            
            // Create model
            auto model = createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            // Model parameters
            int n_components = body.value("n_components", 0);  // 0 = auto
            std::string method = body.value("method", "em");
            int n_iterations = body.value("n_iterations", 100);
            
            // Set model parameters
            if (model_type == "gaussian_mixture" || model_type == "gmm") {
                auto* gmm = dynamic_cast<GaussianMixture*>(model.get());
                gmm->setNumComponents(n_components > 0 ? n_components : 3);
                gmm->setCovarianceType(body.value("covariance_type", "full"));
            } else if (model_type == "dirichlet_process" || model_type == "dp") {
                auto* dp = dynamic_cast<DirichletProcess*>(model.get());
                dp->setConcentration(body.value("alpha", 1.0));
                dp->setBaseMeasure(body.value("base_measure", json::object()));
            } else if (model_type == "hierarchical_dp" || model_type == "hdp") {
                auto* hdp = dynamic_cast<HierarchicalDP*>(model.get());
                hdp->setConcentrations(body.value("gamma", 1.0), body.value("alpha0", 1.0));
            }
            
            // Fit model
            FitResult result;
            if (method == "em") {
                result = model->fitEM(data, n_iterations);
            } else if (method == "gibbs") {
                GibbsSampler sampler;
                result = sampler.fit(model.get(), data, n_iterations);
            } else if (method == "collapsed_gibbs") {
                CollapsedGibbs sampler;
                result = sampler.fit(model.get(), data, n_iterations);
            } else if (method == "variational") {
                result = model->fitVariational(data, n_iterations);
            } else {
                return make_error_response(http::status::bad_request, 
                    "Unknown inference method: " + method, version, keep_alive);
            }
            
            json response = {
                {"model_type", model_type},
                {"method", method},
                {"n_components", result.n_components},
                {"log_likelihood", result.log_likelihood},
                {"bic", result.bic},
                {"aic", result.aic},
                {"converged", result.converged},
                {"iterations", result.iterations},
                {"weights", result.weights}
            };
            
            // Add component parameters
            if (!result.means.empty()) {
                response["means"] = ClusteringUtils::vectorsToJson(result.means);
            }
            if (!result.covariances.empty()) {
                response["covariances"] = ClusteringUtils::matricesToJson(result.covariances);
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Predict/cluster: /mixture/{model_type}/predict
    std::regex predict_regex("/mixture/([a-z_]+)/predict");
    if (std::regex_match(target, match, predict_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data") || !body.contains("parameters")) {
                return make_error_response(http::status::bad_request, 
                    "Missing required fields", version, keep_alive);
            }
            
            // Parse data
            Eigen::MatrixXd data = ClusteringUtils::jsonToMatrix(body["data"]);
            
            // Create and configure model
            auto model = createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            model->setParameters(body["parameters"]);
            
            // Prediction type
            std::string pred_type = body.value("type", "hard");
            
            json response = {
                {"model_type", model_type},
                {"type", pred_type}
            };
            
            if (pred_type == "hard") {
                // Hard clustering
                auto assignments = model->predict(data);
                response["assignments"] = assignments;
                
                // Compute cluster sizes
                std::map<int, int> cluster_sizes;
                for (int label : assignments) {
                    cluster_sizes[label]++;
                }
                response["cluster_sizes"] = cluster_sizes;
                
            } else if (pred_type == "soft") {
                // Soft clustering (probabilities)
                auto probabilities = model->predictProba(data);
                response["probabilities"] = ClusteringUtils::matrixToJson(probabilities);
                
            } else if (pred_type == "density") {
                // Density estimation
                auto densities = model->computeDensity(data);
                response["densities"] = densities;
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Sample from model: /mixture/{model_type}/sample
    std::regex sample_regex("/mixture/([a-z_]+)/sample");
    if (std::regex_match(target, match, sample_regex) && req.method() == http::verb::post) {
        std::string model_type = match[1];
        
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("parameters") || !body.contains("n_samples")) {
                return make_error_response(http::status::bad_request, 
                    "Missing required fields", version, keep_alive);
            }
            
            // Create and configure model
            auto model = createModel(model_type);
            if (!model) {
                return make_error_response(http::status::bad_request, 
                    "Unknown model type: " + model_type, version, keep_alive);
            }
            
            model->setParameters(body["parameters"]);
            
            int n_samples = body["n_samples"];
            bool return_components = body.value("return_components", false);
            
            // Generate samples
            auto sample_result = model->sample(n_samples, return_components);
            
            json response = {
                {"model_type", model_type},
                {"n_samples", n_samples},
                {"samples", ClusteringUtils::matrixToJson(sample_result.samples)}
            };
            
            if (return_components) {
                response["component_labels"] = sample_result.component_labels;
            }
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Model selection: /mixture/select
    if (target == "/mixture/select" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data")) {
                return make_error_response(http::status::bad_request, 
                    "Missing 'data' field", version, keep_alive);
            }
            
            Eigen::MatrixXd data = ClusteringUtils::jsonToMatrix(body["data"]);
            std::vector<std::string> models = body.value("models", 
                std::vector<std::string>{"gaussian_mixture", "dirichlet_process"});
            std::string criterion = body.value("criterion", "bic");
            int max_components = body.value("max_components", 10);
            
            json results = json::array();
            
            for (const auto& model_type : models) {
                auto model = createModel(model_type);
                if (!model) continue;
                
                // Try different number of components
                for (int k = 1; k <= max_components; ++k) {
                    if (model_type == "gaussian_mixture") {
                        auto* gmm = dynamic_cast<GaussianMixture*>(model.get());
                        gmm->setNumComponents(k);
                    }
                    
                    auto fit_result = model->fitEM(data, 100);
                    
                    json model_result = {
                        {"model_type", model_type},
                        {"n_components", k},
                        {"log_likelihood", fit_result.log_likelihood},
                        {"bic", fit_result.bic},
                        {"aic", fit_result.aic}
                    };
                    
                    results.push_back(model_result);
                }
            }
            
            // Sort by criterion
            if (criterion == "bic") {
                std::sort(results.begin(), results.end(), 
                    [](const json& a, const json& b) {
                        return a["bic"] < b["bic"];
                    });
            } else if (criterion == "aic") {
                std::sort(results.begin(), results.end(), 
                    [](const json& a, const json& b) {
                        return a["aic"] < b["aic"];
                    });
            }
            
            json response = {
                {"criterion", criterion},
                {"best_model", results[0]},
                {"all_results", results}
            };
            
            return make_success_response(response, version, keep_alive);
            
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Clustering metrics: /mixture/metrics
    if (target == "/mixture/metrics" && req.method() == http::verb::post) {
        try {
            json body = json::parse(req.body());
            
            if (!body.contains("data") || !body.contains("labels")) {
                return make_error_response(http::status::bad_request, 
                    "Missing required fields", version, keep_alive);
            }
            
            Eigen::MatrixXd data = ClusteringUtils::jsonToMatrix(body["data"]);
            std::vector<int> labels = body["labels"].get<std::vector<int>>();
            
            // Compute various clustering metrics
            double silhouette = ClusteringUtils::silhouetteScore(data, labels);
            double davies_bouldin = ClusteringUtils::daviesBouldinIndex(data, labels);
            double calinski_harabasz = ClusteringUtils::calinskiHarabaszScore(data, labels);
            
            json response = {
                {"silhouette_score", silhouette},
                {"davies_bouldin_index", davies_bouldin},
                {"calinski_harabasz_score", calinski_harabasz}
            };
            
            // If true labels provided, compute supervised metrics
            if (body.contains("true_labels")) {
                std::vector<int> true_labels = body["true_labels"].get<std::vector<int>>();
                
                double ari = ClusteringUtils::adjustedRandIndex(labels, true_labels);
                double nmi = ClusteringUtils::normalizedMutualInfo(labels, true_labels);
                double homogeneity = ClusteringUtils::homogeneityScore(labels, true_labels);
                double completeness = ClusteringUtils::completenessScore(labels, true_labels);
                
                response["adjusted_rand_index"] = ari;
                response["normalized_mutual_info"] = nmi;
                response["homogeneity_score"] = homogeneity;
                response["completeness_score"] = completeness;
                response["v_measure"] = 2 * homogeneity * completeness / (homogeneity + completeness);
            }
            
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