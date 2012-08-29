generate_sourcelists("./")

add_subdirectory(interfaces)
add_subdirectory(performancetests)
add_subdirectory(reversetimemigration)
add_subdirectory(wavefront)

set(RELATIVE_PATH "")
include(auto.cmake)

if(MACHINE_ARCH MATCHES "x86_64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DMY_SIZE=7512")
  add_executable(testbed ${SOURCES})
  target_link_libraries(testbed ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()

