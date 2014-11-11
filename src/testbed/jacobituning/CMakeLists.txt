generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CUDA AND WITH_THREADS)
  cuda_add_executable(jacobituning main.cu)
  target_link_libraries(jacobituning ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
