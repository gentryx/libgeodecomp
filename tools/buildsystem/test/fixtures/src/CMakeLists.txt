cmake_minimum_required(VERSION 2.6)
project(DEMO)

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_FLAGS "-Wall -Werror")
include(conf.cmake)

set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/include/${PACKAGE_NAME}")
include_directories (${DEMO_SOURCE_DIR})

add_definitions(-DDOOMSDAY)

add_subdirectory(testlib)
add_subdirectory(mpilayer)
add_subdirectory(testexe)

set(RELATIVE_PATH testlib/)
include(testlib/auto.cmake)
set(RELATIVE_PATH mpilayer/)
include(mpilayer/auto.cmake)
add_library(superlib SHARED ${SOURCES})
install(TARGETS superlib DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
