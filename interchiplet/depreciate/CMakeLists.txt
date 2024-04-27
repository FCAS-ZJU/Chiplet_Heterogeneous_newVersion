
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
find_package(cppzmq)
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

# Executable: record_transfer
add_executable(record_transfer srcs/record_transfer.cpp)
target_link_libraries(record_transfer ${Boost_LIBRARIES})

# Executable: comm_bridge
add_executable(comm_bridge srcs/comm_bridge.cpp)
target_compile_options(comm_bridge PUBLIC "-pthread")
target_link_options(comm_bridge PUBLIC "-pthread")
#target_link_libraries(comm_bridge ${Boost_LIBRARIES})

# Executable: zmq_pro
# Require cppzmq and Boost. If not found, skip.
if (CPPZMQ_FOUND AND Boost_FOUND)
    add_executable(zmq_pro
        srcs/zmq_pro.cpp
        srcs/interchiplet_client.cpp)

    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(zmq_pro cppzmq)
    target_link_libraries(zmq_pro ${Boost_LIBRARIES})
else()
    if (CPPZMQ_FOUND)
        message(STATUS "Cannot find Boost library, skip target zmq_pro.")
    elseif(Boost_FOUND)
        message(STATUS "Cannot find cppzmq library, skip target zmq_pro.")
    endif()
endif()
