#ifndef STATS_HANDLER_HPP
#define STATS_HANDLER_HPP

#include <boost/beast/http.hpp>

namespace http = boost::beast::http;

http::response<http::string_body> handle_stats_request(
    http::request<http::string_body>&& req);

#endif // STATS_HANDLER_HPP