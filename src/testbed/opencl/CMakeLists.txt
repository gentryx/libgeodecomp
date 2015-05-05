generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_OPENCL)
  add_executable(libgeodecomp_testbed_opencl main.cpp)
  set_target_properties(libgeodecomp_testbed_opencl PROPERTIES OUTPUT_NAME opencl)
  target_link_libraries(libgeodecomp_testbed_opencl ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
