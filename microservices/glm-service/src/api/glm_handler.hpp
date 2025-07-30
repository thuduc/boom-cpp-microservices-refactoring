#ifndef GLM_HANDLER_HPP
#define GLM_HANDLER_HPP

#include <boost/beast/http.hpp>
#include <string>

namespace http = boost::beast::http;

// Handle HTTP requests for GLM service
http::response<http::string_body> handle_glm_request(
    http::request<http::string_body>&& req);

#endif // GLM_HANDLER_HPP