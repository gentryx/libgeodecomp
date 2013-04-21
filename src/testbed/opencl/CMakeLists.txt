generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_OPENCL)
  add_executable(opencl main.cpp)
  target_link_libraries(opencl ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
