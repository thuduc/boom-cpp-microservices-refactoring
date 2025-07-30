#ifndef GIBBS_SAMPLER_HPP
#define GIBBS_SAMPLER_HPP

#include "../models/mixture_base.hpp"

class GibbsSampler {
public:
    FitResult fit(MixtureModel* model, const Eigen::MatrixXd& data, int n_iterations);
    
private:
    void sampleAssignments(MixtureModel* model, const Eigen::MatrixXd& data,
                          std::vector<int>& assignments);
    void updateParameters(MixtureModel* model, const Eigen::MatrixXd& data,
                         const std::vector<int>& assignments);
};