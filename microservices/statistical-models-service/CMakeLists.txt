cmake_minimum_required(VERSION 3.16)
project(statistical_models_service)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${EIGEN3_INCLUDE_DIR})

# Add executable
add_executable(statistical_models_service
    src/main.cpp
    src/api/models_handler.cpp
    src/core/model_factory.cpp
    src/core/models/gaussian_model.cpp
    src/core/models/gamma_model.cpp
    src/core/models/beta_model.cpp
    src/core/models/poisson_model.cpp
    src/core/models/multinomial_model.cpp
    src/core/models/model_base.cpp
    src/core/estimation/mle_estimator.cpp
    src/core/estimation/bayesian_estimator.cpp
    src/utils/data_validator.cpp
    src/utils/statistics_utils.cpp
)

# Link libraries
target_link_libraries(statistical_models_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)