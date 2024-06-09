
cmake_minimum_required(VERSION 3.10)
enable_language(CUDA)

# Project name.
project(interchiplet LANGUAGES CUDA CXX)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  #GIT_TAG <tag_name> # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
  message(STATUS "Fetching yaml-cpp...")
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

# Parameter set.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
include_directories(includes)

# Import library
# find cppzmq wrapper, installed by make of cppzmq
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
    program_options
)

# Static library: interchiplet_c
add_library(interchiplet_c STATIC
    srcs/apis_c.cpp)

# Static library: interchiplet_cu
# add_library(interchiplet_cu STATIC
#     srcs/apis_cu.cu)

# Executable: interchiplet
add_executable(interchiplet srcs/interchiplet.cpp)
target_compile_options(interchiplet PUBLIC "-pthread")
target_link_options(interchiplet PUBLIC "-pthread")
target_link_libraries(interchiplet ${Boost_LIBRARIES})
target_link_libraries(interchiplet yaml-cpp::yaml-cpp) # The library or executable that require yaml-cpp library
