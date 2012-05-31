generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(MACHINE_ARCH MATCHES "x86_64")
  set(CMAKE_CXX_FLAGS "-g -DMY_SIZE=7512")

  add_executable(interfaces ${SOURCES})
  target_link_libraries(interfaces ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
