generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if((MACHINE_ARCH MATCHES "x86_64") AND WITH_CUDA)
  # set(CMAKE_CXX_FLAGS "-g")

  add_executable(rtm main.cpp)
  # cuda_add_executable(rtm ${SOURCES})

  cuda_add_library(cudaStepper cudastepperlib.cu)

  target_link_libraries(rtm cudaStepper)
  target_link_libraries(rtm ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
