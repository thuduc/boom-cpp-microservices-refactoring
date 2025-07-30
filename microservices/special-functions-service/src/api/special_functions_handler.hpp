#ifndef SPECIAL_FUNCTIONS_HANDLER_HPP
#define SPECIAL_FUNCTIONS_HANDLER_HPP

#include <boost/beast/http.hpp>

namespace http = boost::beast::http;

http::response<http::string_body> handle_special_functions_request(
    http::request<http::string_body>&& req);

#endif // SPECIAL_FUNCTIONS_HANDLER_HPP