# 项目信息
project (zmq_pro)
# 指定生成目标
add_executable(zmq_pro zmq_pro.cpp interchiplet_client.cpp)

set(CMAKE_CXX_STANDARD 17)

#find cppzmq wrapper, installed by make of cppzmq
find_package(cppzmq)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(zmq_pro cppzmq)
target_link_libraries(zmq_pro ${Boost_LIBRARIES})

#记录转换
add_executable(record_transfer record_transfer.cpp)
target_link_libraries(record_transfer ${Boost_LIBRARIES})
