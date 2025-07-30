#ifndef TS_HANDLER_HPP
#define TS_HANDLER_HPP

#include <boost/beast/http.hpp>

namespace http = boost::beast::http;

http::response<http::string_body> handle_ts_request(
    http::request<http::string_body>&& req);

#endif // TS_HANDLER_HPP