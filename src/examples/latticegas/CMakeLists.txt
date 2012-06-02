generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if (FEATURE_CUDA AND FEATURE_QT AND FEATURE_OPENCV)
  include(${QT_USE_FILE})

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

  # fixme: rework this
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

  # set(CUDA_NVCC_FLAGS "-arch sm_20 ${CUDA_NVCC_FLAGS}")u
  cuda_add_executable(latticegas ${SOURCES} ${MY_MOC_SOURCES})
  target_link_libraries(latticegas ${LOCAL_LIBGEODECOMP_LINK_LIB} ${QT_LIBRARIES} ${OpenCV_LIBS})
endif()
