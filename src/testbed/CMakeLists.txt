set(RELATIVE_PATH "")
include(auto.cmake)

if (LIBGEODECOMP_FEATURE_OPENCL)
  add_executable(testbed ${SOURCES})
  target_link_libraries(testbed ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(LIBGEODECOMP_FEATURE_OPENCL)
