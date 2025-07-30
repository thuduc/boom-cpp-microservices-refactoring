#ifndef CHAIN_MANAGER_HPP
#define CHAIN_MANAGER_HPP

#include <string>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <chrono>
#include "../core/mcmc_engine.hpp"

struct ChainInfo {
    std::string chain_id;
    std::string method;
    std::string status; // "running", "completed", "failed"
    std::string created_at;
    std::string updated_at;
    MCMCResult result;
    std::string error_message;
};

class ChainManager {
public:
    ChainManager();
    
    // Create a new chain
    std::string createChain(const std::string& method);
    
    // Update chain status
    void completeChain(const std::string& chain_id, const MCMCResult& result);
    void failChain(const std::string& chain_id, const std::string& error_message);
    
    // Get chain information
    std::optional<ChainInfo> getChainInfo(const std::string& chain_id) const;
    
    // Get number of active chains
    size_t getActiveChainCount() const;
    
    // Clean up old chains
    void cleanupOldChains(std::chrono::hours age_limit = std::chrono::hours(24));
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ChainInfo> chains_;
    
    std::string generateChainId() const;
    std::string getCurrentTimestamp() const;
};

#endif // CHAIN_MANAGER_HPP