generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if (WITH_CUDA AND WITH_QT AND WITH_OPENCV)
  include(${QT_USE_FILE})

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

  set(MY_HEADERS
    cameratester.h
    flowwidget.h
    framegrabber.h
    interactivesimulator.h
    interactivesimulatorcpu.h
    interactivesimulatorgpu.h
    )
  set(MY_UI)
  set(MY_RESOURCES)

  QT4_WRAP_CPP(MY_MOC_SOURCES ${MY_HEADERS} OPTIONS -DBOOST_TT_HAS_OPERATOR_HPP_INCLUDED)
  QT4_WRAP_UI(MY_UI_SOURCES ${MY_UI})
  QT4_ADD_RESOURCES(MY_RESOURCE_HEADERS ${MY_RESOURCES})
  include_directories(./)
  include_directories(${OpenCV_INCLUDE_DIRS})

  cuda_add_executable(libgeodecomp_examples_latticegas ${SOURCES} ${MY_MOC_SOURCES})
  set_target_properties(libgeodecomp_examples_latticegas PROPERTIES OUTPUT_NAME latticegas)
  target_link_libraries(libgeodecomp_examples_latticegas ${LOCAL_LIBGEODECOMP_LINK_LIB} ${QT_LIBRARIES} ${OpenCV_LIBS})
endif()
