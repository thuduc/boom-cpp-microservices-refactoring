#ifndef MODEL_FACTORY_HPP
#define MODEL_FACTORY_HPP

#include <memory>
#include <string>
#include "models/model_base.hpp"

class ModelFactory {
public:
    // Create a model instance by type name
    static std::unique_ptr<StatisticalModel> createModel(const std::string& model_type);
};

#endif // MODEL_FACTORY_HPP