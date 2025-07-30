#include "model_base.hpp"

double StatisticalModel::negativeLLogLikelihood(const std::vector<double>& params, 
                                              const std::vector<double>& data,
                                              StatisticalModel* model) {
    // Convert params vector to json for the model
    json param_json;
    // This will be overridden by specific models
    
    model->setParameters(param_json);
    return -model->logLikelihood(data);
}