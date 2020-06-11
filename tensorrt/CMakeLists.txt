
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

find_package(GStreamer)
if(GStreamer_FOUND)
  message(STATUS "FOUND!!")
  include_directories(${GSTREAMER_INCLUDE_DIRS})
endif()

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
        /usr/lib/x86_64-linux-gnu/glib-2.0/include
        /usr/include/glib-2.0
        /usr/local/include/gstreamer-1.0
        /usr/local/include/libsoup-2.4
        /usr/local/include/json-glib-1.0) 

link_directories( ${INCLUDE_DIR} ${CMAKE_PREFIX_PATH} ${GSTLIBS_LIBRARY_DIRS})

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

target_link_libraries(
        ${PROJECT_NAME} 
        ${CUDA_LIBRARIES} 
        ${INCLUDE_DIR} 
        ${NVINFER} 
        ${NVPARSERS} 
        ${NVONNXPARSERS} 
        ${NVPLUGIN}
        ${GSTLIBS_LIBRARIES})



add_executable(unet_model trtexec.cpp)
target_link_libraries(unet_model ${PROJECT_NAME})


set(SOURCE_FILES_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/main.cpp)
set(SOURCE_FILES_WEBRTC ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/rtsp_webrtc.cpp)
set(SOURCE_FILES_RTSP ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/rtsp_restream_text.cpp)
set(SOURCE_FILES_RTP_TEST ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/gst_rtp_test.cpp)
# set(SOURCE_FILES_RTSP_APPSRC ${CMAKE_CURRENT_SOURCE_DIR}/gstreamer/rtsp_stream_appsrc.cpp)


add_executable(mainapp ${SOURCE_FILES_MAIN})
add_executable(rtsp2webrtc ${SOURCE_FILES_WEBRTC})
add_executable(rtsprestream ${SOURCE_FILES_RTSP})
add_executable(gstrtptest ${SOURCE_FILES_RTP_TEST})
# add_executable(rtspstreamappsrc ${SOURCE_FILES_RTSP_APPSRC})


target_link_libraries(mainapp ${GSTREAMER_LIBRARIES})
target_link_libraries(rtsp2webrtc ${GSTREAMER_LIBRARIES})
target_link_libraries(rtsprestream ${GSTREAMER_LIBRARIES})
target_link_libraries(gstrtptest ${GSTREAMER_LIBRARIES})
# target_link_libraries(rtspstreamappsrc ${GSTREAMER_LIBRARIES})
