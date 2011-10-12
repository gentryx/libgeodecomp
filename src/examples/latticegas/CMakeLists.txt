set(RELATIVE_PATH "")
include(auto.cmake)

find_package(CUDA)

# fixme: use configure script to find cuda
if (CUDA_FOUND)
  cuda_add_executable(latticegas ${SOURCES})
  target_link_libraries(latticegas ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif(CUDA_FOUND)
