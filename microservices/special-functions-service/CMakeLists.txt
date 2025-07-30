cmake_minimum_required(VERSION 3.16)
project(special_functions_service)

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
    src/api/special_functions_handler.cpp
    src/core/bessel_functions.cpp
    src/core/gamma_functions.cpp
    src/core/beta_functions.cpp
    src/core/fft_calculator.cpp
    src/core/polygamma_functions.cpp
    src/core/hypergeometric_functions.cpp
    src/core/elliptic_functions.cpp
    src/utils/numeric_utils.cpp
)

# Create executable
add_executable(special_functions_service ${SOURCES})

# Link libraries
target_link_libraries(special_functions_service
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    pthread
)

# Set output directory
set_target_properties(special_functions_service PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)