cmake_minimum_required(VERSION 3.23)
project(BP_OF_DL_CPP)

set(CMAKE_CXX_STANDARD 14)
include_directories( "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3" )
add_executable(BP_OF_DL_CPP
        DL_CPP_TEST_1_22281E114_JONYAN.cpp)
