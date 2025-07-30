cmake_minimum_required(VERSION 3.16)
project(mcmc_service)

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
add_executable(mcmc_service
    src/main.cpp
    src/api/mcmc_handler.cpp
    src/core/mcmc_engine.cpp
    src/core/samplers/metropolis_hastings.cpp
    src/core/samplers/slice_sampler.cpp
    src/core/samplers/arms_sampler.cpp
    src/core/samplers/gibbs_sampler.cpp
    src/core/chain_diagnostics.cpp
    src/models/chain_manager.cpp
    src/utils/distribution_parser.cpp
)

# Link libraries
target_link_libraries(mcmc_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)