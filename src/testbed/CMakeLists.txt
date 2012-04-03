set(RELATIVE_PATH "")
include(auto.cmake)

set(CMAKE_CXX_FLAGS "-O3 -march=native -g")
#if (LIBGEODECOMP_FEATURE_OPENCL)
  add_executable(testbed ${SOURCES})
  target_link_libraries(testbed ${LOCAL_LIBGEODECOMP_LINK_LIB})
#  target_link_libraries(testbed OpenCL)
#endif(LIBGEODECOMP_FEATURE_OPENCL)
