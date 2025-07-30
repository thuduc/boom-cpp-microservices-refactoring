cmake_minimum_required(VERSION 3.10)
project(TimeSeriesService)

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
    src/api/ts_handler.cpp
    src/core/models/arima_model.cpp
    src/core/models/local_level.cpp
    src/core/models/local_linear_trend.cpp
    src/core/models/seasonal.cpp
    src/core/filters/kalman_filter.cpp
    src/core/forecasters/forecaster_base.cpp
    src/utils/time_series_utils.cpp
)

# Create executable
add_executable(time_series_service ${SOURCES})

# Link libraries
target_link_libraries(time_series_service
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
)

# Set output name
set_target_properties(time_series_service PROPERTIES OUTPUT_NAME "time_series_service")