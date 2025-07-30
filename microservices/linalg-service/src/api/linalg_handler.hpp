#ifndef LINALG_HANDLER_HPP
#define LINALG_HANDLER_HPP

#include <boost/beast/http.hpp>
#include <string>

namespace http = boost::beast::http;

// Handle HTTP requests for Linear Algebra service
http::response<http::string_body> handle_linalg_request(http::request<http::string_body>&& req);

#endif // LINALG_HANDLER_HPP