cmake_minimum_required(VERSION 3.10)
project(MixtureModelsService)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/main.cpp
    src/api/mixture_handler.cpp
    src/core/models/mixture_base.cpp
    src/core/models/gaussian_mixture.cpp
    src/core/models/dirichlet_process.cpp
    src/core/models/hierarchical_dp.cpp
    src/core/samplers/gibbs_sampler.cpp
    src/core/samplers/collapsed_gibbs.cpp
    src/utils/clustering_utils.cpp
)

# Create executable
add_executable(mixture_models_service ${SOURCES})

# Link libraries
target_link_libraries(mixture_models_service
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

# Set output name
set_target_properties(mixture_models_service PROPERTIES OUTPUT_NAME "mixture_models_service")