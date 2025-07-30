#ifndef MODELS_HANDLER_HPP
#define MODELS_HANDLER_HPP

#include <boost/beast/http.hpp>
#include <string>

namespace http = boost::beast::http;

// Handle HTTP requests for Statistical Models service
http::response<http::string_body> handle_models_request(
    http::request<http::string_body>&& req);

#endif // MODELS_HANDLER_HPP