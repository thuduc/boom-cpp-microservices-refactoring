#include "chain_manager.hpp"
#include <random>
#include <sstream>
#include <iomanip>
#include <ctime>

ChainManager::ChainManager() {}

std::string ChainManager::generateChainId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    const char* hex_chars = "0123456789abcdef";
    std::string chain_id = "chain-";
    
    for (int i = 0; i < 16; ++i) {
        chain_id += hex_chars[dis(gen)];
        if (i == 3 || i == 7 || i == 11) {
            chain_id += '-';
        }
    }
    
    return chain_id;
}

std::string ChainManager::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::string ChainManager::createChain(const std::string& method) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string chain_id = generateChainId();
    ChainInfo& chain = chains_[chain_id];
    
    chain.chain_id = chain_id;
    chain.method = method;
    chain.status = "running";
    chain.created_at = getCurrentTimestamp();
    chain.updated_at = chain.created_at;
    
    return chain_id;
}

void ChainManager::completeChain(const std::string& chain_id, const MCMCResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = chains_.find(chain_id);
    if (it != chains_.end()) {
        it->second.status = "completed";
        it->second.result = result;
        it->second.updated_at = getCurrentTimestamp();
    }
}

void ChainManager::failChain(const std::string& chain_id, const std::string& error_message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = chains_.find(chain_id);
    if (it != chains_.end()) {
        it->second.status = "failed";
        it->second.error_message = error_message;
        it->second.updated_at = getCurrentTimestamp();
    }
}

std::optional<ChainInfo> ChainManager::getChainInfo(const std::string& chain_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = chains_.find(chain_id);
    if (it != chains_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

size_t ChainManager::getActiveChainCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = 0;
    for (const auto& [id, chain] : chains_) {
        if (chain.status == "running") {
            count++;
        }
    }
    
    return count;
}

void ChainManager::cleanupOldChains(std::chrono::hours age_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto cutoff = now - age_limit;
    
    for (auto it = chains_.begin(); it != chains_.end(); ) {
        if (it->second.status != "running") {
            // Parse timestamp and check age
            std::tm tm = {};
            std::istringstream ss(it->second.updated_at);
            ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
            
            auto chain_time = std::chrono::system_clock::from_time_t(std::mktime(&tm));
            
            if (chain_time < cutoff) {
                it = chains_.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}