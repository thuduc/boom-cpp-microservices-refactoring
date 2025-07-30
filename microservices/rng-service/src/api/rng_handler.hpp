#ifndef RNG_HANDLER_HPP
#define RNG_HANDLER_HPP

#include <boost/beast/http.hpp>
#include <string>

namespace http = boost::beast::http;

// Handle HTTP requests for RNG service
http::response<http::string_body> handle_request(http::request<http::string_body>&& req);

#endif // RNG_HANDLER_HPP