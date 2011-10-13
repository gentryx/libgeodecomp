set(RELATIVE_PATH "")
include(auto.cmake)

find_package(CUDA)
find_package(Qt4)

# fixme: use configure script to find cuda and Qt4
if (CUDA_FOUND)
  include(${QT_USE_FILE})

  set(CMAKE_CXX_FLAGS "-I/usr/include/opencv  -lml -lcvaux -lhighgui -lcv -lcxcore ${CMAKE_CXX_FLAGS}")

  # fixme: rework this
  set(MY_HEADERS
    flowwidget.h
    framegrabber.h
    interactivesimulator.h
    )
  set(MY_UI)
  set(MY_RESOURCES)
  
  QT4_WRAP_CPP(MY_MOC_SOURCES ${MY_HEADERS})
  QT4_WRAP_UI(MY_UI_SOURCES ${MY_UI})
  QT4_ADD_RESOURCES(MY_RESOURCE_HEADERS ${MY_RESOURCES})
  include_directories(./)

  cuda_add_executable(latticegas ${SOURCES} ${MY_MOC_SOURCES})
  target_link_libraries(latticegas ${LOCAL_LIBGEODECOMP_LINK_LIB} ${QT_LIBRARIES})
endif(CUDA_FOUND)
