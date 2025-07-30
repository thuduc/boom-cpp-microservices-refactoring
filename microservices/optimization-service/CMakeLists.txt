cmake_minimum_required(VERSION 3.16)
project(optimization_service)

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
add_executable(optimization_service
    src/main.cpp
    src/api/optimization_handler.cpp
    src/core/optimization_engine.cpp
    src/core/optimizers/bfgs_optimizer.cpp
    src/core/optimizers/nelder_mead_optimizer.cpp
    src/core/optimizers/powell_optimizer.cpp
    src/core/optimizers/simulated_annealing.cpp
    src/core/optimizers/newton_optimizer.cpp
    src/models/job_manager.cpp
    src/utils/function_parser.cpp
)

# Link libraries
target_link_libraries(optimization_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)