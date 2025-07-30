#ifndef JOB_MANAGER_HPP
#define JOB_MANAGER_HPP

#include <string>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <chrono>
#include "../core/optimization_engine.hpp"

struct JobInfo {
    std::string job_id;
    std::string method;
    std::string status; // "running", "completed", "failed"
    std::string created_at;
    std::string updated_at;
    OptimizationResult result;
    std::string error_message;
};

class JobManager {
public:
    JobManager();
    
    // Create a new job
    std::string createJob(const std::string& method);
    
    // Update job status
    void completeJob(const std::string& job_id, const OptimizationResult& result);
    void failJob(const std::string& job_id, const std::string& error_message);
    
    // Get job information
    std::optional<JobInfo> getJobInfo(const std::string& job_id) const;
    
    // Get number of active jobs
    size_t getActiveJobCount() const;
    
    // Clean up old jobs (optional)
    void cleanupOldJobs(std::chrono::hours age_limit = std::chrono::hours(24));
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, JobInfo> jobs_;
    
    std::string generateJobId() const;
    std::string getCurrentTimestamp() const;
};

#endif // JOB_MANAGER_HPP