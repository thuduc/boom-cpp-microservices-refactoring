#ifndef MCMC_HANDLER_HPP
#define MCMC_HANDLER_HPP

#include <boost/beast/http.hpp>
#include <memory>
#include <string>

namespace http = boost::beast::http;

class ChainManager;

// Handle HTTP requests for MCMC service
http::response<http::string_body> handle_mcmc_request(
    http::request<http::string_body>&& req,
    std::shared_ptr<ChainManager> chain_manager);

#endif // MCMC_HANDLER_HPP