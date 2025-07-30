#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/config.hpp>
#include "api/mcmc_handler.hpp"
#include "models/chain_manager.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

// Global chain manager
std::shared_ptr<ChainManager> g_chain_manager;

// HTTP session handler
void do_session(tcp::socket socket) {
    bool close = false;
    beast::error_code ec;
    beast::flat_buffer buffer;
    
    // Increase buffer size for MCMC data
    buffer.max_size(20 * 1024 * 1024); // 20MB
    
    while (!close) {
        // Read request
        http::request<http::string_body> req;
        http::read(socket, buffer, req, ec);
        if (ec == http::error::end_of_stream) break;
        if (ec) return;
        
        // Handle request
        http::response<http::string_body> res = handle_mcmc_request(
            std::move(req), g_chain_manager);
        res.set(http::field::server, "BOOM MCMC Sampling Service/1.0");
        res.prepare_payload();
        
        // Send response
        http::write(socket, res, ec);
        if (ec) return;
        
        close = res.need_eof();
    }
    
    socket.shutdown(tcp::socket::shutdown_send, ec);
}

// Accept incoming connections
void do_accept(net::io_context& ioc, tcp::acceptor& acceptor) {
    acceptor.async_accept(
        [&](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                std::thread{std::bind(&do_session, std::move(socket))}.detach();
            }
            do_accept(ioc, acceptor);
        });
}

int main(int argc, char* argv[]) {
    try {
        // Check command line arguments
        if (argc != 3) {
            std::cerr << "Usage: mcmc_service <address> <port>\n";
            std::cerr << "  Example: mcmc_service 0.0.0.0 8083\n";
            return 1;
        }
        
        auto const address = net::ip::make_address(argv[1]);
        auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
        
        // Initialize chain manager
        g_chain_manager = std::make_shared<ChainManager>();
        
        // IO context
        net::io_context ioc{1};
        
        // Acceptor
        tcp::acceptor acceptor{ioc, {address, port}};
        
        // Start accepting connections
        do_accept(ioc, acceptor);
        
        std::cout << "MCMC Sampling Service listening on " << address << ":" << port << std::endl;
        
        // Run the IO context
        ioc.run();
    }
    catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}