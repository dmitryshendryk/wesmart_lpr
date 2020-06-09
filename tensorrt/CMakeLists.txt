
cmake_minimum_required(VERSION 3.16)


project(trt)


set(CMAKE_PREFIX_PATH "/home/wesmart/Documents/apps/TensorRT-7.0.0.11/lib/") 
set(INCLUDE_DIR "/home/wesmart/Documents/apps/TensorRT-7.0.0.11/samples/common/") 


find_library(NVINFER NAMES libnvinfer.so)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
find_package(CUDA  REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories (${INCLUDE_DIR}) 

link_directories(${INCLUDE_DIR})

if(NVINFER)
   message("TensorRT is available!")
   message("NVINFER: ${NVINFER}")
   message("NVPARSERS: ${NVPARSERS}")
   message("NVONNXPARSERS: ${NVONNXPARSERS}")
   set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

SET(SAMPLE_SOURCES
    trtexec.cpp
)

add_executable(trtexec trtexec.cpp)

target_link_libraries(trt ${INCLUDE_DIR} )