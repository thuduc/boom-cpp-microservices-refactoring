#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

// Forward declaration
http::response<http::string_body> handle_ts_request(http::request<http::string_body>&& req);

// Session class to handle each connection
class session : public std::enable_shared_from_this<session> {
    tcp::socket socket_;
    beast::flat_buffer buffer_;
    http::request<http::string_body> req_;

public:
    explicit session(tcp::socket socket) : socket_(std::move(socket)) {}

    void run() {
        read_request();
    }

private:
    void read_request() {
        auto self = shared_from_this();

        http::async_read(socket_, buffer_, req_,
            [self](beast::error_code ec, std::size_t) {
                if (!ec) {
                    self->process_request();
                }
            });
    }

    void process_request() {
        auto response = handle_ts_request(std::move(req_));
        
        auto self = shared_from_this();
        http::async_write(socket_, response,
            [self](beast::error_code ec, std::size_t) {
                self->socket_.shutdown(tcp::socket::shutdown_send, ec);
            });
    }
};

// Listener class to accept connections
class listener : public std::enable_shared_from_this<listener> {
    net::io_context& ioc_;
    tcp::acceptor acceptor_;

public:
    listener(net::io_context& ioc, tcp::endpoint endpoint)
        : ioc_(ioc)
        , acceptor_(net::make_strand(ioc)) {
        beast::error_code ec;

        acceptor_.open(endpoint.protocol(), ec);
        if (ec) {
            std::cerr << "Error opening acceptor: " << ec.message() << std::endl;
            return;
        }

        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        if (ec) {
            std::cerr << "Error setting socket option: " << ec.message() << std::endl;
            return;
        }

        acceptor_.bind(endpoint, ec);
        if (ec) {
            std::cerr << "Error binding: " << ec.message() << std::endl;
            return;
        }

        acceptor_.listen(net::socket_base::max_listen_connections, ec);
        if (ec) {
            std::cerr << "Error listening: " << ec.message() << std::endl;
            return;
        }
    }

    void run() {
        accept();
    }

private:
    void accept() {
        acceptor_.async_accept(
            net::make_strand(ioc_),
            beast::bind_front_handler(&listener::on_accept, shared_from_this()));
    }

    void on_accept(beast::error_code ec, tcp::socket socket) {
        if (!ec) {
            std::make_shared<session>(std::move(socket))->run();
        }
        accept();
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: time_series_service <address> <port>\n";
        std::cerr << "Example: time_series_service 0.0.0.0 8086\n";
        return 1;
    }

    auto const address = net::ip::make_address(argv[1]);
    auto const port = static_cast<unsigned short>(std::atoi(argv[2]));

    net::io_context ioc{std::thread::hardware_concurrency()};

    std::make_shared<listener>(ioc, tcp::endpoint{address, port})->run();

    std::cout << "Time Series Service listening on " << address << ":" << port << std::endl;

    std::vector<std::thread> threads;
    threads.reserve(std::thread::hardware_concurrency() - 1);
    for (auto i = std::thread::hardware_concurrency() - 1; i > 0; --i) {
        threads.emplace_back([&ioc] {
            ioc.run();
        });
    }

    ioc.run();

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}