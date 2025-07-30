cmake_minimum_required(VERSION 3.16)
project(rng_service)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add executable
add_executable(rng_service
    src/main.cpp
    src/api/rng_handler.cpp
    src/core/rng_engine.cpp
    src/core/distribution_factory.cpp
    src/utils/json_utils.cpp
)

# Link libraries
target_link_libraries(rng_service
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
)

# Copy necessary BOOM source files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../distributions/rng.cpp 
               ${CMAKE_CURRENT_BINARY_DIR}/rng.cpp COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../distributions/rng.hpp 
               ${CMAKE_CURRENT_BINARY_DIR}/rng.hpp COPYONLY)