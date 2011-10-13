set(RELATIVE_PATH "")
include(auto.cmake)

find_package(CUDA)
find_package(Qt4)

# fixme: use configure script to find cuda and Qt4
if (CUDA_FOUND)
  include(${QT_USE_FILE})

  # fixme: rework this
  SET(MY_HEADERS
    flowwidget.h
    framegrabber.h
    interactivesimulator.h
    )
  SET(MY_UI)
  SET(MY_RESOURCES)
  
  QT4_WRAP_CPP(MY_MOC_SOURCES ${MY_HEADERS})
  QT4_WRAP_UI(MY_UI_SOURCES ${MY_UI})
  QT4_ADD_RESOURCES(MY_RESOURCE_HEADERS ${MY_RESOURCES})
  INCLUDE_DIRECTORIES( . )

  
  cuda_add_executable(latticegas ${SOURCES} ${MY_MOC_SOURCES})
  target_link_libraries(latticegas ${LOCAL_LIBGEODECOMP_LINK_LIB} ${QT_LIBRARIES})
endif(CUDA_FOUND)
