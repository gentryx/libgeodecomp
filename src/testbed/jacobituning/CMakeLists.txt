lgd_generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CUDA AND WITH_THREADS AND WITH_CPP14)
  lfa_cuda_add_executable(libgeodecomp_testbed_jacobituning main.cu)
  set_target_properties(libgeodecomp_testbed_jacobituning PROPERTIES OUTPUT_NAME jacobituning)
  target_link_libraries(libgeodecomp_testbed_jacobituning ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
