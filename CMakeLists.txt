cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(PACKAGE_NAME libgeodecomp)
set(CMAKE_C_FLAGS   "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${ADDITIONAL_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

set(PACKAGE_VERSION "0.3.0")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http:\\\\www.libgeodecomp.org")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(util)
include(CheckCXXCompilerFlag)

add_subdirectory(src)
