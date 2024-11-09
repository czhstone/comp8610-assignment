CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
PROJECT(rope_simulation)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

add_library(rope_lib SHARED
    src/transforms.cpp
    src/rasterizer.cpp
    src/rope.cpp
    src/cloth.cpp
)

target_link_libraries(rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task1 task1.cpp)
target_link_libraries(task1 rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task2_1 task2_1.cpp)
target_link_libraries(task2_1 rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task2_2 task2_2.cpp)
target_link_libraries(task2_2 rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task3 task3.cpp)
target_link_libraries(task3 rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task4 task4.cpp)
target_link_libraries(task4 rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task5_1 task5_1.cpp)
target_link_libraries(task5_1 rope_lib ${OpenCV_LIBS})

ADD_EXECUTABLE(task5_2 task5_2.cpp)
target_link_libraries(task5_2 rope_lib ${OpenCV_LIBS})
