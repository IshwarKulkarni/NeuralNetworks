cmake_minimum_required(VERSION 3.7)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 11)

set( SRC "src/")
include_directories("src/")

add_definitions( -DDATA_LOCATION="../../data")

set(SOURCE_FILES
        ${SRC}data/CIFAR-ImageReader.hxx
        ${SRC}data/DataSets.cxx
        ${SRC}data/DataSets.hxx
        ${SRC}data/MNIST-ImageReader.cxx
        ${SRC}data/MNIST-ImageReader.hxx

        ${SRC}neuralnets/Activation.hxx
        ${SRC}neuralnets/AttenuationLayer.hxx
        ${SRC}neuralnets/AveragePoolingLayer.hxx
        ${SRC}neuralnets/ConvolutionLayer.hxx
        ${SRC}neuralnets/DropConnectLayer.hxx
        ${SRC}neuralnets/ErrorFunctions.hxx
        ${SRC}neuralnets/FullyConnectedLayer.hxx
        ${SRC}neuralnets/Layer.hxx
        ${SRC}neuralnets/MaxPoolingLayer.hxx
        ${SRC}neuralnets/Network.cxx
        ${SRC}neuralnets/Network.hxx
        ${SRC}neuralnets/NeuralNet_Main.cxx

        ${SRC}utils/CommandLine.hxx
        ${SRC}utils/PPMIO.cxx
        ${SRC}utils/SimpleMatrix.hxx
        ${SRC}utils/Utils.hxx
        ${SRC}utils/Vec23.hxx)

add_executable(NeuralNet ${SOURCE_FILES} )

target_link_libraries(NeuralNet "pthread")
