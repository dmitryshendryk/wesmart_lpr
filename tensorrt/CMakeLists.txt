
cmake_minimum_required(VERSION 3.16)

SET(SAMPLE_SOURCES
    ../../TensorRT-7.0.0.11/samples/common/sampleEngines.cpp
    ../../TensorRT-7.0.0.11/samples/common/sampleInference.cpp
    ../../TensorRT-7.0.0.11/samples/common/sampleOptions.cpp
    ../../TensorRT-7.0.0.11/samples/common/sampleReporting.cpp
    trtexec.cpp
)

set(SAMPLE_PARSERS "caffe" "uff" "onnx")
set(PLUGINS_NEEDED ON)
include(../../CMakeSamplesTemplate.txt)