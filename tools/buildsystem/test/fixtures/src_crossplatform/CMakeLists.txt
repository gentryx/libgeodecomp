cmake_minimum_required(VERSION 2.6)
project(DEMO)

set(CMAKE_VERBOSE_MAKEFILE TRUE)
include(conf.cmake)

set(INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/include/${PACKAGE_NAME}")
include_directories (${DEMO_SOURCE_DIR})

add_definitions(-DDOOMSDAY)

add_subdirectory(testlib)
add_subdirectory(testexe)

set(RELATIVE_PATH testlib/)
include(testlib/auto.cmake)
add_library(superlib STATIC ${SOURCES})
install(TARGETS superlib DESTINATION lib)
