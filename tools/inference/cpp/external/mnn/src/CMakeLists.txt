cmake_minimum_required(VERSION 3.0)

#set the project name and vesion
set(PROJECT external_exec)
project(${PROJECT} VERSION 1.0)

#public lib path
set(OPENCV_PATH /usr/local/opencv_op)
set(MNN_PATH /usr/local/mnn)

#add compile option
if(CMAKE_BUILD_TYPE STREQUAL "release")
    message(STATUS "build release...")
    #add_compile_options(-O2 -Wall)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -o2 -fopenmp -mavx2 --std=c++17")
else()
    message(STATUS "build debug...")
    #add_compile_options(-g -O2 -Wall)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -fopenmp -mavx2 -std=c++17")
endif()

#specify the c++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

#add the executable
list(APPEND SRC_DIRS ${PROJECT_SOURCE_DIR})

foreach (dir ${SRC_DIRS})
    aux_source_directory(${dir} dir_files)
    set(SOURCE ${SOURCE} ${dir_files})
endforeach ()

# file(GLOB_RECURSE SOURCE "*.cpp")
# file(GLOB_RECURSE REMOVE "${PROJECT_SOURCE_DIR}/build/*")
# list(REMOVE_ITEM SOURCE ${REMOVE})
message(STATUS "find compile files ${SOURCE}")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OPENCV_PATH}/include/opencv4)
include_directories(${ROOT_PATH})
include_directories(${MNN_PATH}/include)
link_directories(${OPENCV_PATH}/lib)
link_directories(${MNN_PATH}/lib)

add_executable(${PROJECT} ${SOURCE})

#add lib
set(OPENCV_LIB_ITEM opencv_imgproc opencv_highgui opencv_core opencv_imgcodecs opencv_dnn)
set(MNN_LIB_ITEM MNN)
target_link_libraries(${PROJECT} ${OPENCV_LIB_ITEM} ${MNN_LIB_ITEM})


