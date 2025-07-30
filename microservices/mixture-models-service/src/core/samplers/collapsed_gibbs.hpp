#ifndef COLLAPSED_GIBBS_HPP
#define COLLAPSED_GIBBS_HPP

#include "../models/mixture_base.hpp"

class CollapsedGibbs {
public:
    FitResult fit(MixtureModel* model, const Eigen::MatrixXd& data, int n_iterations);
    
private:
    double computeMarginalLikelihood(const Eigen::MatrixXd& cluster_data) const;
};