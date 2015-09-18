lgd_generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_INTRINSICS)
  add_executable(libgeodecomp_testbed_performancetests main.cpp)
  set_target_properties(libgeodecomp_testbed_performancetests PROPERTIES OUTPUT_NAME performancetests)
  target_link_libraries(libgeodecomp_testbed_performancetests ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()

if(WITH_CUDA AND WITH_INTRINSICS)
  cuda_add_library(libgeodecomp_testbed_performancetests_cudatests cudatests.cu)
  target_link_libraries(libgeodecomp_testbed_performancetests libgeodecomp_testbed_performancetests_cudatests)
endif()
