#ifndef OPTIMIZATION_HANDLER_HPP
#define OPTIMIZATION_HANDLER_HPP

#include <boost/beast/http.hpp>
#include <memory>
#include <string>

namespace http = boost::beast::http;

class JobManager;

// Handle HTTP requests for Optimization service
http::response<http::string_body> handle_optimization_request(
    http::request<http::string_body>&& req,
    std::shared_ptr<JobManager> job_manager);

#endif // OPTIMIZATION_HANDLER_HPP