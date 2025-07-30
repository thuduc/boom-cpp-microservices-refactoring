cmake_minimum_required(VERSION 3.16)
project(glm_service)

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
add_executable(glm_service
    src/main.cpp
    src/api/glm_handler.cpp
    src/core/glm_base.cpp
    src/core/models/linear_regression.cpp
    src/core/models/logistic_regression.cpp
    src/core/models/poisson_regression.cpp
    src/core/models/probit_regression.cpp
    src/core/estimation/irls_solver.cpp
    src/core/estimation/gradient_solver.cpp
    src/core/diagnostics/regression_diagnostics.cpp
    src/utils/data_preprocessor.cpp
    src/utils/link_functions.cpp
)

# Link libraries
target_link_libraries(glm_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)