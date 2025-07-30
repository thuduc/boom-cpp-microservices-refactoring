#include "job_manager.hpp"
#include <random>
#include <sstream>
#include <iomanip>
#include <ctime>

JobManager::JobManager() {}

std::string JobManager::generateJobId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    const char* hex_chars = "0123456789abcdef";
    std::string job_id = "job-";
    
    for (int i = 0; i < 16; ++i) {
        job_id += hex_chars[dis(gen)];
        if (i == 3 || i == 7 || i == 11) {
            job_id += '-';
        }
    }
    
    return job_id;
}

std::string JobManager::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::string JobManager::createJob(const std::string& method) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string job_id = generateJobId();
    JobInfo& job = jobs_[job_id];
    
    job.job_id = job_id;
    job.method = method;
    job.status = "running";
    job.created_at = getCurrentTimestamp();
    job.updated_at = job.created_at;
    
    return job_id;
}

void JobManager::completeJob(const std::string& job_id, const OptimizationResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = jobs_.find(job_id);
    if (it != jobs_.end()) {
        it->second.status = "completed";
        it->second.result = result;
        it->second.updated_at = getCurrentTimestamp();
    }
}

void JobManager::failJob(const std::string& job_id, const std::string& error_message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = jobs_.find(job_id);
    if (it != jobs_.end()) {
        it->second.status = "failed";
        it->second.error_message = error_message;
        it->second.updated_at = getCurrentTimestamp();
    }
}

std::optional<JobInfo> JobManager::getJobInfo(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = jobs_.find(job_id);
    if (it != jobs_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

size_t JobManager::getActiveJobCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t count = 0;
    for (const auto& [id, job] : jobs_) {
        if (job.status == "running") {
            count++;
        }
    }
    
    return count;
}

void JobManager::cleanupOldJobs(std::chrono::hours age_limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto now = std::chrono::system_clock::now();
    auto cutoff = now - age_limit;
    
    for (auto it = jobs_.begin(); it != jobs_.end(); ) {
        if (it->second.status != "running") {
            // Parse timestamp and check age
            std::tm tm = {};
            std::istringstream ss(it->second.updated_at);
            ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
            
            auto job_time = std::chrono::system_clock::from_time_t(std::mktime(&tm));
            
            if (job_time < cutoff) {
                it = jobs_.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}