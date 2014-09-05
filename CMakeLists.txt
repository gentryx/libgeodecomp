cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(PACKAGE_NAME libgeodecomp)
set(PACKAGE_VERSION "0.3.1")
set(PACKAGE_VENDOR "Chair for Computer Science 3, FAU Erlangen, Germany")
set(PACKAGE_HOMEPAGE "http:\\\\www.libgeodecomp.org")
set(PACKAGE_EMAIL "users@libgeodecomp.org")
project(${PACKAGE_NAME})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
include(util)
include(CheckCXXCompilerFlag)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_subdirectory(src)

# export this for users who bundle a checkout of LibGeoDecomp in their own CMake source tree
set(LIBGEODECOMP_ALL_INCLUDE_DIRECTORIES ${LIBGEODECOMP_ALL_INCLUDE_DIRECTORIES} PARENT_SCOPE)
