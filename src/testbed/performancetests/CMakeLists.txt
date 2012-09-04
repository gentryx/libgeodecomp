generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(MACHINE_ARCH MATCHES "x86_64")
  add_executable(performancetests ${SOURCES})
  target_link_libraries(performancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
