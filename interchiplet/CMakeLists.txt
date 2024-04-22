
cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)

# Project name.
project(interchiplet LANGUAGES CUDA CXX)

# Parameter set.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(includes)

# Import library
# find cppzmq wrapper, installed by make of cppzmq
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)

# Static library: interchiplet_c
add_library(interchiplet_c STATIC
    srcs/apis_c.cpp)

# Static library: interchiplet_cu
add_library(interchiplet_cu STATIC
    srcs/apis_cu.cu)

# Executable: interchiplet
add_executable(interchiplet srcs/interchiplet.cpp)
target_compile_options(interchiplet PUBLIC "-pthread")
target_link_options(interchiplet PUBLIC "-pthread")
#target_link_libraries(interchiplet ${Boost_LIBRARIES})
