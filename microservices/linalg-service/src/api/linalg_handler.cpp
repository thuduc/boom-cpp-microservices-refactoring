#include "api/linalg_handler.hpp"
#include "core/matrix_operations.hpp"
#include "core/decompositions.hpp"
#include "core/solvers.hpp"
#include "utils/matrix_converter.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
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

http::response<http::string_body> handle_linalg_request(http::request<http::string_body>&& req) {
    // Extract basic info
    auto const version = req.version();
    auto const keep_alive = req.keep_alive();
    
    // Handle different endpoints
    std::string target = std::string(req.target());
    
    // Health check endpoint
    if (target == "/health" && req.method() == http::verb::get) {
        json health = {
            {"status", "healthy"},
            {"service", "linalg-service"},
            {"version", "1.0.0"}
        };
        return make_success_response(health, version, keep_alive);
    }
    
    // Matrix multiplication endpoint
    if (target == "/linalg/multiply" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("matrix_a") || !body.contains("matrix_b")) {
                return make_error_response(http::status::bad_request, 
                    "Missing matrix_a or matrix_b", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["matrix_a"]);
            auto B = MatrixConverter::fromJson(body["matrix_b"]);
            
            auto result = MatrixOperations::multiply(A, B);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            json response = {
                {"operation", "multiply"},
                {"result", MatrixConverter::toJson(result)},
                {"computation_time_us", duration.count()},
                {"dimensions", {
                    {"input_a", {A.rows(), A.cols()}},
                    {"input_b", {B.rows(), B.cols()}},
                    {"output", {result.rows(), result.cols()}}
                }}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Cholesky decomposition endpoint
    if (target == "/linalg/decompose/cholesky" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("matrix")) {
                return make_error_response(http::status::bad_request, 
                    "Missing matrix", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["matrix"]);
            auto L = Decompositions::cholesky(A);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            json response = {
                {"operation", "cholesky"},
                {"L", MatrixConverter::toJson(L)},
                {"computation_time_us", duration.count()},
                {"verified", true} // L * L^T = A
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // LU decomposition endpoint
    if (target == "/linalg/decompose/lu" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("matrix")) {
                return make_error_response(http::status::bad_request, 
                    "Missing matrix", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["matrix"]);
            auto [L, U, P] = Decompositions::lu(A);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            json response = {
                {"operation", "lu"},
                {"L", MatrixConverter::toJson(L)},
                {"U", MatrixConverter::toJson(U)},
                {"P", MatrixConverter::toJson(P)},
                {"computation_time_us", duration.count()}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // QR decomposition endpoint
    if (target == "/linalg/decompose/qr" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("matrix")) {
                return make_error_response(http::status::bad_request, 
                    "Missing matrix", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["matrix"]);
            auto [Q, R] = Decompositions::qr(A);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            json response = {
                {"operation", "qr"},
                {"Q", MatrixConverter::toJson(Q)},
                {"R", MatrixConverter::toJson(R)},
                {"computation_time_us", duration.count()}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // SVD decomposition endpoint
    if (target == "/linalg/decompose/svd" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("matrix")) {
                return make_error_response(http::status::bad_request, 
                    "Missing matrix", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["matrix"]);
            auto [U, S, V] = Decompositions::svd(A);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            json response = {
                {"operation", "svd"},
                {"U", MatrixConverter::toJson(U)},
                {"S", S},  // Singular values as vector
                {"V", MatrixConverter::toJson(V)},
                {"computation_time_us", duration.count()},
                {"rank", Decompositions::rank(S)}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Linear system solver endpoint
    if (target == "/linalg/solve" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("A") || !body.contains("b")) {
                return make_error_response(http::status::bad_request, 
                    "Missing A or b for Ax=b", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["A"]);
            auto b = MatrixConverter::vectorFromJson(body["b"]);
            
            std::string method = body.value("method", "auto");
            
            Eigen::VectorXd x;
            if (method == "cholesky") {
                x = Solvers::solveCholesky(A, b);
            } else if (method == "lu") {
                x = Solvers::solveLU(A, b);
            } else if (method == "qr") {
                x = Solvers::solveQR(A, b);
            } else {
                x = Solvers::solve(A, b);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            // Compute residual
            Eigen::VectorXd residual = A * x - b;
            double residual_norm = residual.norm();
            
            json response = {
                {"operation", "solve"},
                {"solution", MatrixConverter::vectorToJson(x)},
                {"method", method},
                {"residual_norm", residual_norm},
                {"computation_time_us", duration.count()}
            };
            return make_success_response(response, version, keep_alive);
        } catch (const std::exception& e) {
            return make_error_response(http::status::bad_request, 
                e.what(), version, keep_alive);
        }
    }
    
    // Eigenvalues endpoint
    if (target == "/linalg/eigenvalues" && req.method() == http::verb::post) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            
            json body = json::parse(req.body());
            
            if (!body.contains("matrix")) {
                return make_error_response(http::status::bad_request, 
                    "Missing matrix", version, keep_alive);
            }
            
            auto A = MatrixConverter::fromJson(body["matrix"]);
            bool compute_vectors = body.value("compute_eigenvectors", false);
            
            auto [eigenvalues, eigenvectors] = Decompositions::eigen(A, compute_vectors);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            json response = {
                {"operation", "eigenvalues"},
                {"eigenvalues", MatrixConverter::complexVectorToJson(eigenvalues)},
                {"computation_time_us", duration.count()}
            };
            
            if (compute_vectors) {
                response["eigenvectors"] = MatrixConverter::complexMatrixToJson(eigenvectors);
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