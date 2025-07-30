cmake_minimum_required(VERSION 3.16)
project(linalg_service)

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
add_executable(linalg_service
    src/main.cpp
    src/api/linalg_handler.cpp
    src/core/matrix_operations.cpp
    src/core/decompositions.cpp
    src/core/solvers.cpp
    src/utils/matrix_converter.cpp
)

# Link libraries
target_link_libraries(linalg_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)