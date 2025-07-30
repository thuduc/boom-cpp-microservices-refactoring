#include "model_factory.hpp"
#include "models/gaussian_model.hpp"
#include "models/gamma_model.hpp"
#include "models/beta_model.hpp"
#include "models/poisson_model.hpp"
#include "models/multinomial_model.hpp"

std::unique_ptr<StatisticalModel> ModelFactory::createModel(const std::string& model_type) {
    if (model_type == "gaussian" || model_type == "normal") {
        return std::make_unique<GaussianModel>();
    } else if (model_type == "gamma") {
        return std::make_unique<GammaModel>();
    } else if (model_type == "beta") {
        return std::make_unique<BetaModel>();
    } else if (model_type == "poisson") {
        return std::make_unique<PoissonModel>();
    } else if (model_type == "multinomial") {
        return std::make_unique<MultinomialModel>();
    }
    
    return nullptr;
}