cmake_minimum_required(VERSION 3.16)
project(statistical_utilities_service)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(SOURCES
    src/main.cpp
    src/api/stats_handler.cpp
    src/core/descriptive_stats.cpp
    src/core/hypothesis_tests.cpp
    src/core/correlation_analysis.cpp
    src/core/resampling_methods.cpp
    src/core/distribution_fitting.cpp
    src/core/time_series_analysis.cpp
    src/core/multivariate_analysis.cpp
    src/utils/data_transformer.cpp
)

# Create executable
add_executable(statistical_utilities_service ${SOURCES})

# Link libraries
target_link_libraries(statistical_utilities_service
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    pthread
)

# Set output directory
set_target_properties(statistical_utilities_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)