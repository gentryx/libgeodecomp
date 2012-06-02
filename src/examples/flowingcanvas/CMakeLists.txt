generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_CUDA AND FEATURE_QT AND FEATURE_OPENCV)
  include(${QT_USE_FILE})

  # fixme: rework this
  set(MY_HEADERS
    canvaswriter.h
    flowwidget.h
    framegrabber.h
    interactivesimulator.h)

  QT4_WRAP_CPP(MY_MOC_SOURCES ${MY_HEADERS})
  include_directories(./)
  include_directories(${OpenCV_INCLUDE_DIRS})

  cuda_add_executable(flowingcanvas ${SOURCES} ${MY_MOC_SOURCES})
  target_link_libraries(flowingcanvas ${LOCAL_LIBGEODECOMP_LINK_LIB} ${QT_LIBRARIES} ${OpenCV_LIBRARIES})
endif()
