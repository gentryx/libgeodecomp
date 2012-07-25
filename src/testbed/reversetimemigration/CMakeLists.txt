generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(MACHINE_ARCH MATCHES "x86_64")
  # set(CMAKE_CXX_FLAGS "-g")

  add_executable(rtm ${SOURCES})
  target_link_libraries(rtm ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
