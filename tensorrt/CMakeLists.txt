
cmake_minimum_required(VERSION 3.16)


project(trt)


set(CMAKE_PREFIX_PATH "/home/wesmart/Documents/apps/TensorRT-7.0.0.11/lib/") 
set(INCLUDE_DIR "/home/wesmart/Documents/apps/TensorRT-7.0.0.11/samples/common/") 
set(HEADER_FILES ${INCLUDE_DIR}/logger.h ${INCLUDE_DIR}/buffers.h ${INCLUDE_DIR}/common.h)

find_library(NVINFER NAMES libnvinfer.so)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
find_library(NVPLUGIN NAMES libnvinfer_plugin.so)
find_package(CUDA  REQUIRED)

include_directories (${INCLUDE_DIR} ${CMAKE_PREFIX_PATH} ${CUDA_INCLUDE_DIRS}) 

link_directories( ${INCLUDE_DIR} ${CMAKE_PREFIX_PATH})
#add_subdirectory(${INCLUDE_DIR})

if(NVINFER)
   message("TensorRT is available!")
   message("NVINFER: ${NVINFER}")
   message("NVPARSERS: ${NVPARSERS}")
   message("NVONNXPARSERS: ${NVONNXPARSERS}")
   message("NVPLUGIN: ${NVPLUGIN}")
   set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

add_library(${PROJECT_NAME} SHARED 
  ${INCLUDE_DIR}/logger.cpp
  ${INCLUDE_DIR}/logger.h
  ${INCLUDE_DIR}/buffers.h
  ${INCLUDE_DIR}/common.h
  ${INCLUDE_DIR}/sampleEngines.cpp
  ${INCLUDE_DIR}/sampleInference.cpp
  ${INCLUDE_DIR}/sampleReporting.cpp
  ${INCLUDE_DIR}/sampleOptions.cpp

)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${INCLUDE_DIR} ${NVINFER} ${NVPARSERS} ${NVONNXPARSERS} ${NVPLUGIN})



add_executable( unet_model
              trtexec.cpp)
target_link_libraries(unet_model ${PROJECT_NAME})