
cmake_minimum_required(VERSION 3.10)


project(trt)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)


set(CMAKE_PREFIX_PATH "/usr/lib/aarch64-linux-gnu") 
set(INCLUDE_DIR "/usr/src/tensorrt/samples/common/") 
set(HEADER_FILES ${INCLUDE_DIR}/logger.h ${INCLUDE_DIR}/buffers.h ${INCLUDE_DIR}/common.h)

find_library(NVINFER NAMES libnvinfer.so)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)
find_library(NVPLUGIN NAMES libnvinfer_plugin.so)

find_package(CUDA  REQUIRED)
find_package(OpenCV REQUIRED)

# SET ( CMAKE_CXX_FLAGS "--expt-extended-lambda" CACHE STRING "compile flags" FORCE)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS};-std=c++11;--expt-extended-lambda;)

# list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
# add nvcc flags to generate code for all enabled architectures

enable_language(CUDA)

# list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# enable device lambda support
# xxxnsubtil: not usable yet
# list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

# set optimization flags
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTLIBS REQUIRED
        gobject-2.0
        glib-2.0
        gstreamer-webrtc-1.0
        gstreamer-sdp-1.0
        gstreamer-pbutils-1.0
        libsoup-2.4
        json-glib-1.0
        gstreamer-rtsp-server-1.0
        gstreamer-check-1.0)

include_directories (
        ${INCLUDE_DIR} 
        ${CMAKE_PREFIX_PATH} 
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GSTLIBS_LIBRARY_DIRS}
        /usr/lib/aarch64-linux-gnu/glib-2.0/include
        /usr/include/glib-2.0
        /usr/include/gstreamer-1.0
        /usr/include/libsoup-2.4
        /usr/include/json-glib-1.0) 

# link_directories( ${INCLUDE_DIR} ${CMAKE_PREFIX_PATH} )

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

# add_library(${PROJECT_NAME} SHARED 
#   ${INCLUDE_DIR}/logger.cpp
#   ${INCLUDE_DIR}/logger.h
#   ${INCLUDE_DIR}/buffers.h
#   ${INCLUDE_DIR}/common.h
#   ${INCLUDE_DIR}/sampleEngines.cpp
#   ${INCLUDE_DIR}/sampleInference.cpp
#   ${INCLUDE_DIR}/sampleReporting.cpp
#   ${INCLUDE_DIR}/sampleOptions.cpp

# )

add_library(${PROJECT_NAME} SHARED
  # ./csrc/cuda/decode.h
  # ./csrc/cuda/decode.cu
  # ./csrc/cuda/nms.h
  # ./csrc/cuda/nms.cu
  # ./csrc/cuda/decode_rotate.h
  # ./csrc/cuda/decode_rotate.cu
  # ./csrc/cuda/nms_iou.h
  # ./csrc/cuda/nms_iou.cu
  ./csrc/cuda/utils.h
  ./csrc/engine.h
  ./csrc/engine.cpp
  ./csrc/calibrator.h
  ${INCLUDE_DIR}/logger.cpp
  ${INCLUDE_DIR}/logger.h
  ${INCLUDE_DIR}/buffers.h
  ${INCLUDE_DIR}/common.h
  ${INCLUDE_DIR}/sampleEngines.cpp
  ${INCLUDE_DIR}/sampleInference.cpp
  ${INCLUDE_DIR}/sampleReporting.cpp
  ${INCLUDE_DIR}/sampleOptions.cpp
)

# set_target_properties(trt_engine PROPERTIES
#   CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )


# target_link_libraries(
#         ${PROJECT_NAME} 
#         ${CUDA_LIBRARIES} 
#         ${INCLUDE_DIR} 
#         ${NVINFER} 
#         ${NVPARSERS} 
#         ${NVONNXPARSERS} 
#         ${NVPLUGIN}
#         ${GSTLIBS_LIBRARIES}
#         ${OpenCV_LIBS}
#         trt_engine)



add_executable(unet_model trtexec.cpp)
target_link_libraries(unet_model 
        ${PROJECT_NAME}
        ${CUDA_LIBRARIES} 
        ${INCLUDE_DIR} 
        ${NVINFER} 
        ${NVPARSERS} 
        ${NVONNXPARSERS} 
        ${NVPLUGIN}
        ${GSTLIBS_LIBRARIES}
        ${OpenCV_LIBS})

add_executable(infer_model inference.cpp)
target_link_libraries(infer_model 
        ${PROJECT_NAME}
        ${CUDA_LIBRARIES} 
        ${INCLUDE_DIR} 
        ${NVINFER} 
        ${NVPARSERS} 
        ${NVONNXPARSERS} 
        ${NVPLUGIN}
        ${GSTLIBS_LIBRARIES}
        ${OpenCV_LIBS})
  
add_executable(infervideo infervideo.cpp)
target_link_libraries(infervideo 
        ${PROJECT_NAME}
        ${CUDA_LIBRARIES} 
        ${INCLUDE_DIR} 
        ${NVINFER} 
        ${NVPARSERS} 
        ${NVONNXPARSERS} 
        ${NVPLUGIN}
        ${GSTLIBS_LIBRARIES}
        ${OpenCV_LIBS})

set(SOURCE_FILES_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/main.cpp)
set(SOURCE_FILES_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/gst_video.cpp)
set(SOURCE_FILES_WEBRTC ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/rtsp_webrtc.cpp)
set(SOURCE_FILES_RTSP ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/rtsp_restream_text.cpp)
set(SOURCE_FILES_RTP_TEST ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/gst_rtp_test.cpp)
set(SOURCE_FILES_RTSP_APPSRC ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/rtsp_stream_appsrc.cpp)


add_executable(mainapp ${SOURCE_FILES_MAIN})
add_executable(run_video ${SOURCE_FILES_MAIN})
add_executable(rtsp2webrtc ${SOURCE_FILES_WEBRTC})
add_executable(rtsprestream ${SOURCE_FILES_RTSP})
add_executable(gstrtptest ${SOURCE_FILES_RTP_TEST})
add_executable(rtspstreamappsrc ${SOURCE_FILES_RTSP_APPSRC})


target_link_libraries(mainapp ${GSTLIBS_LIBRARIES})
target_link_libraries(run_video ${GSTLIBS_LIBRARIES})
target_link_libraries(rtsp2webrtc ${GSTLIBS_LIBRARIES})
target_link_libraries(rtsprestream ${GSTLIBS_LIBRARIES})
target_link_libraries(gstrtptest ${GSTLIBS_LIBRARIES})
target_link_libraries(rtspstreamappsrc ${GSTLIBS_LIBRARIES})
