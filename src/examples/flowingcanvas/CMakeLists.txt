set(RELATIVE_PATH "")
include(auto.cmake)

find_package(Qt4)

# fixme: use configure script to find opencv, same in latticegas
execute_process(COMMAND pkg-config opencv --cflags --libs
  OUTPUT_VARIABLE OPENCV_FLAGS)
STRING(REGEX REPLACE "\n" "" OPENCV_FLAGS ${OPENCV_FLAGS})
set(CMAKE_CXX_FLAGS "${OPENCV_FLAGS} ${CMAKE_CXX_FLAGS} -fopenmp")

include(${QT_USE_FILE})

set(MY_HEADERS
  flowwidget.h
  interactivesimulator.h)

QT4_WRAP_CPP(MY_MOC_SOURCES ${MY_HEADERS})
include_directories(./)

add_executable(flowingcanvas ${SOURCES} ${MY_MOC_SOURCES})
target_link_libraries(flowingcanvas ${LOCAL_LIBGEODECOMP_LINK_LIB})
