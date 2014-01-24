generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_OPENCL)
  add_executable(opencl main.cpp)
  target_link_libraries(opencl ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
