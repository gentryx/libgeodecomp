generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_INTRINSICS)
  add_executable(performancetests main.cpp)
  target_link_libraries(performancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()

if(WITH_CUDA AND WITH_INTRINSICS)
  cuda_add_library(cudatests cudatests.cu)
  target_link_libraries(performancetests cudatests)
endif()
