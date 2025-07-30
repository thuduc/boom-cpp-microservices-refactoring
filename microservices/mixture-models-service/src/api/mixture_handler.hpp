#ifndef MIXTURE_HANDLER_HPP
#define MIXTURE_HANDLER_HPP

#include <boost/beast/http.hpp>

namespace http = boost::beast::http;

http::response<http::string_body> handle_mixture_request(
    http::request<http::string_body>&& req);

#endif // MIXTURE_HANDLER_HPP